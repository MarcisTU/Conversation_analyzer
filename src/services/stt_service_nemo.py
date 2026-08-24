import argparse
import asyncio
import json
import os
import sys
from pathlib import Path
from typing import Tuple, List, Optional

from dataclasses import dataclass, field
import librosa
import torch
import numpy as np
import torchaudio
from dotenv import load_dotenv
from loguru import logger

import lightning.pytorch as pl
from omegaconf import OmegaConf

from nemo.collections.asr.metrics.wer import word_error_rate
from nemo.collections.asr.parts.submodules.ctc_decoding import CTCDecodingConfig
from nemo.collections.asr.parts.submodules.rnnt_decoding import RNNTDecodingConfig
from nemo.collections.asr.parts.utils.manifest_utils import read_manifest
from nemo.collections.asr.parts.utils.rnnt_utils import Hypothesis
from nemo.collections.asr.parts.utils.streaming_utils import CacheAwareStreamingAudioBuffer
from nemo.collections.asr.parts.utils.transcribe_utils import get_inference_device, get_inference_dtype, setup_model
from nemo.core.config import hydra_runner
from nemo.utils import logging

from src.modules.constants import ROOT_DIR, SRC_DIR
from src.models.enums import SegmentType
from src.models.results import Results
from src.models.results import ResultsSegment
from src.services.base_service import BaseService
from src.utils.audio_utils import AudioUtils


@dataclass
class TranscriptionConfig:
    """
    Transcription Configuration for cache-aware inference.
    """

    # Required configs
    model_path: Optional[str] = None  # Path to a .nemo file
    pretrained_name: Optional[str] = "nvidia/nemotron-3.5-asr-streaming-0.6b"  # Name of a pretrained model
    audio_dir: Optional[str] = None  # Path to a directory which contains audio files
    audio_type: str = "wav"  # type of audio file if audio_dir passed
    audio_file: Optional[str] = None  # Path to an audio file to perform streaming
    dataset_manifest: Optional[str] = None  # Path to dataset's JSON manifest
    output_path: Optional[str] = None  # Path to output file when manifest is used as input

    # General configs
    batch_size: int = 32
    # num_workers: int = 0
    # append_pred: bool = False  # Sets mode of work, if True it will add new field transcriptions.
    # pred_name_postfix: Optional[str] = None  # If you need to use another model name, rather than standard one.
    random_seed: Optional[int] = None  # seed number going to be used in seed_everything()

    # Chunked configs
    chunk_size: int = -1  # The chunk_size to be used for models trained with full context and offline models
    shift_size: int = -1  # The shift_size to be used for models trained with full context and offline models
    left_chunks: Optional[int] = (
        2  # The number of left chunks to be used as left context via caching for offline models
    )
    online_normalization: bool = False  # Perform normalization on the run per chunk.
    # `pad_and_drop_preencoded` enables padding the audio input and then dropping the extra steps after
    # the pre-encoding for all the steps including the the first step. It may make the outputs of the downsampling
    # slightly different from offline mode for some techniques like striding or sw_striding.
    pad_and_drop_preencoded: bool = False
    att_context_size: Optional[list] = field(default_factory=lambda: [56, 13])  # Sets the att_context_size for the models which support multiple lookaheads

    compare_vs_offline: bool = False  #  Whether to compare the output of the model with the offline mode.

    # Set `cuda` to int to define CUDA device. If 'None', will look for CUDA
    # device anyway, and do inference on CPU only if CUDA device is not found.
    # If `cuda` is a negative number, inference will be on CPU only.
    cuda: Optional[int] = None
    allow_mps: bool = False  # allow to select MPS device (Apple Silicon M-series GPU)
    amp: bool = False
    amp_dtype: str = "float16"  # can be set to "float16" or "bfloat16" when using amp
    # NB: default compute_dtype is float32 since currently cache-aware models do not work with different dtype
    compute_dtype: Optional[str] = (
        "float32"  # "float32" (default), "bfloat16" or "float16"; if None: bfloat16 if available else float32
    )
    matmul_precision: str = "high"  # Literal["highest", "high", "medium"]

    # Decoding strategy for CTC models
    ctc_decoding: CTCDecodingConfig = field(default_factory=CTCDecodingConfig)
    # Decoding strategy for RNNT models
    rnnt_decoding: RNNTDecodingConfig = field(default_factory=lambda: RNNTDecodingConfig(fused_batch_size=-1))
    # Selects the decoder for Hybrid ASR models which has both the CTC and RNNT decoder.
    decoder_type: Optional[str] = "rnnt"  # Literal["ctc", "rnnt"]

    # Config for word / character error rate calculation
    # calculate_wer: bool = True
    # clean_groundtruth_text: bool = False
    # langid: str = "en"  # specify this for convert_num_to_words step in groundtruth cleaning
    # use_cer: bool = False
    debug_mode: bool = False  # Whether to print more detail in the output.

    # Language-ID prompt for prompt-conditioned models (e.g. EncDecRNNTBPEModelWithPrompt).
    # Set to a language key from the model's prompt_dictionary (e.g. "en-US", "auto").
    # Ignored for models without prompt support.
    target_lang: Optional[str] = "en-US"
    # whether to strip the language tags from the transcriptions
    # Ignored for model without prompt support
    strip_lang_tags: bool = True
    # Optional regex describing the language tag to strip. Defaults to "<xx-XX>". (r'\s*<[a-z]{2}-[A-Z]{2}>')
    lang_tag_pattern: Optional[str] = None


# TODO: Finetuning: https://github.com/nvidia-riva/tutorials/blob/main/asr-finetune-nemotron-3.5-asr-streaming-prompt.ipynb
class SttService(BaseService):
    def __init__(self, args, batch_size=2):
        try:
            super().__init__()
            self.args = args
            self.batch_size = batch_size

            try:
                self.cfg = OmegaConf.structured(TranscriptionConfig)

                self.device = get_inference_device(cuda=self.cfg.cuda, allow_mps=self.cfg.allow_mps)
                logger.info(f"Device: {self.device}")

                self.compute_dtype: torch.dtype
                if self.cfg.amp:
                    # with amp model weights required to be in float32
                    self.compute_dtype = torch.float32
                else:
                    self.compute_dtype = get_inference_dtype(compute_dtype=self.cfg.compute_dtype, device=self.device)

                self.asr_model, self.model_name = setup_model(cfg=self.cfg, map_location=self.device)
                logger.info(f"{self.model_name} loaded.")

                logger.info(self.asr_model.encoder.streaming_cfg)
                if self.cfg.att_context_size is not None:
                    if hasattr(self.asr_model.encoder, "set_default_att_context_size"):
                        self.asr_model.encoder.set_default_att_context_size(att_context_size=self.cfg.att_context_size)
                    else:
                        raise ValueError("Model does not support multiple lookaheads.")

                # Setup decoding strategy
                if hasattr(self.asr_model, 'change_decoding_strategy') and hasattr(self.asr_model, 'decoding'):
                    if self.cfg.decoder_type is not None:
                        decoding_cfg = self.cfg.rnnt_decoding if self.cfg.decoder_type == 'rnnt' else self.cfg.ctc_decoding

                        if hasattr(self.asr_model, 'cur_decoder'):
                            self.asr_model.change_decoding_strategy(decoding_cfg, decoder_type=self.cfg.decoder_type)
                        else:
                            self.asr_model.change_decoding_strategy(decoding_cfg)

                    # Check if ctc or rnnt model
                    elif hasattr(self.asr_model, 'joint'):  # RNNT model
                        self.cfg.rnnt_decoding.fused_batch_size = -1
                        if hasattr(self.asr_model, 'cur_decoder'):
                            self.asr_model.change_decoding_strategy(self.cfg.rnnt_decoding, decoder_type=self.cfg.decoder_type)
                        else:
                            self.asr_model.change_decoding_strategy(self.cfg.rnnt_decoding)
                    else:
                        self.asr_model.change_decoding_strategy(self.cfg.ctc_decoding)

                # Set language-ID prompt for prompt-conditioned models
                if hasattr(self.asr_model, 'set_inference_prompt'):
                    lang = self.cfg.target_lang if self.cfg.target_lang is not None else "auto"
                    self.asr_model.set_inference_prompt(lang)
                    self.asr_model.decoding.set_strip_lang_tags(self.cfg.strip_lang_tags, lang_tag_pattern=self.cfg.lang_tag_pattern)

                self.asr_model = self.asr_model.to(device=self.device, dtype=self.compute_dtype)
                self.asr_model.eval()

                if self.cfg.chunk_size > 0:
                    if self.cfg.shift_size < 0:
                        shift_size = self.cfg.chunk_size
                    else:
                        shift_size = self.cfg.shift_size
                    self.asr_model.encoder.setup_streaming_params(
                        chunk_size=self.cfg.chunk_size, left_chunks=self.cfg.left_chunks, shift_size=shift_size
                    )

                # In streaming, offline normalization is not feasible as we don't have access to the whole audio at the beginning
                # When online_normalization is enabled, the normalization of the input features (mel-spectrograms) are done per step
                # It is suggested to train the streaming models without any normalization in the input features.
                if self.cfg.online_normalization:
                    if self.asr_model.cfg.preprocessor.normalize not in ["per_feature", "all_feature"]:
                        logging.warning(
                            "online_normalization is enabled but the model has no normalization in the feature extration part, so it is ignored."
                        )
                        online_normalization = False
                    else:
                        online_normalization = True

                else:
                    online_normalization = False

                self.streaming_buffer = CacheAwareStreamingAudioBuffer(
                    model=self.asr_model,
                    online_normalization=online_normalization,
                    pad_and_drop_preencoded=self.cfg.pad_and_drop_preencoded,
                )

                logger.info("Stt service initialized successfully!")

            except Exception as exc:
                raise FileNotFoundError(f"Error loading audio stt model: {exc}")

        except Exception as exc:
            logger.exception(exc)

    def _extract_transcriptions(self, hyps):
        """
        The transcribed_texts returned by CTC and RNNT models are different.
        This method would extract and return the text section of the hypothesis.
        """
        if isinstance(hyps[0], Hypothesis):
            transcriptions = []
            for hyp in hyps:
                transcriptions.append(hyp.text)
        else:
            transcriptions = hyps
        return transcriptions

    def _perform_streaming(self):
        batch_size = len(self.streaming_buffer.streams_length)

        cache_last_channel, cache_last_time, cache_last_channel_len = self.asr_model.encoder.get_initial_cache_state(
            batch_size=batch_size
        )

        previous_hypotheses = None
        streaming_buffer_iter = iter(self.streaming_buffer)
        pred_out_stream = None
        for step_num, (chunk_audio, chunk_lengths) in enumerate(streaming_buffer_iter):
            with torch.inference_mode():
                # keep_all_outputs needs to be True for the last step of streaming when model is trained with att_context_style=regular
                # otherwise the last outputs would get dropped
                chunk_audio = chunk_audio.to(self.compute_dtype)
                with torch.no_grad():
                    (
                        pred_out_stream,
                        transcribed_texts,
                        cache_last_channel,
                        cache_last_time,
                        cache_last_channel_len,
                        previous_hypotheses,
                    ) = self.asr_model.conformer_stream_step(
                        processed_signal=chunk_audio,
                        processed_signal_length=chunk_lengths,
                        cache_last_channel=cache_last_channel,
                        cache_last_time=cache_last_time,
                        cache_last_channel_len=cache_last_channel_len,
                        keep_all_outputs=self.streaming_buffer.is_buffer_empty(),
                        previous_hypotheses=previous_hypotheses,
                        previous_pred_out=pred_out_stream,
                        drop_extra_pre_encoded=0
                            if step_num == 0 and not self.cfg.pad_and_drop_preencoded
                            else self.asr_model.encoder.streaming_cfg.drop_extra_pre_encoded,
                        return_transcription=True,
                    )

        final_streaming_tran = self._extract_transcriptions(transcribed_texts)
        logging.info(f"Final streaming transcriptions: {final_streaming_tran}")

        self.streaming_buffer.reset_buffer()

        return final_streaming_tran

    async def inference(
        self,
        file_path: str,
        existing_results: Results
    ) -> Tuple[Results, str]:
        """
        Asynchronous wrapper for the stt pipeline.
        This offloads the heavy CUDA/CPU computation to a separate OS thread,
        preventing the asyncio event loop from freezing.
        """
        logger.info(f"Starting async thread for stt inference on: {file_path}")

        return await asyncio.to_thread(
            self._run_inference,
            file_path=file_path,
            existing_results=existing_results
        )

    def _run_inference(
        self,
        file_path: str,
        existing_results: Results
    ) -> Tuple[Results, str]:
        error_message = None
        try:
            sampling_rate = 16000
            wav, sr = librosa.load(file_path, sr=sampling_rate)
            logger.debug(f"Audio total samples: {len(wav)}, shape: {wav.shape}")

            segment_batches = []
            segment_batches_idxs = []
            cur_batch = []
            cur_batch_idxs = []
            for idx_s, segment in enumerate(existing_results.segments):
                if segment.type == SegmentType.speech:
                    start_sample = int(segment.start_time * sampling_rate)
                    end_sample = int(segment.end_time * sampling_rate)
                    wav_part = wav[start_sample:end_sample]

                    cur_batch.append(wav_part)
                    cur_batch_idxs.append(idx_s)

                if len(cur_batch) == self.batch_size:
                    segment_batches.append(cur_batch)
                    segment_batches_idxs.append(cur_batch_idxs)
                    cur_batch = []
                    cur_batch_idxs = []

            if len(cur_batch) <= self.batch_size:
                segment_batches.append(cur_batch)
                segment_batches_idxs.append(cur_batch_idxs)

            for process_batch, process_batch_idxs in zip(segment_batches, segment_batches_idxs):
                for wav_seg in process_batch:
                    _ = self.streaming_buffer.append_audio(audio=wav_seg, stream_id=-1)

                final_streaming_tran = self._perform_streaming()

                for i_r, result_seg_idx in enumerate(process_batch_idxs):
                    existing_results.segments[result_seg_idx].text = final_streaming_tran[i_r]

        except Exception as exc:
            logger.exception(exc)
            if error_message is None:
                error_message = "(type=%r, value=%r, traceback=%r)" % sys.exc_info()

        return existing_results, error_message


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-device', default='cuda', type=str)
    parser.add_argument('-datasource_samplerate', default=16000, type=int)
    args, _ = parser.parse_known_args()

    load_dotenv(f"{ROOT_DIR}/.env")

    stt_service = SttService(args, batch_size=4)

    file_path_input = f"{ROOT_DIR}/tests/JRE_Chase_Hughes_16k_mono.wav"

    with open(f"{ROOT_DIR}/tests/JRE_Chase_Hughes_16k_mono_diarize_result.json", "r", encoding="utf-8") as f:
        json_data = json.load(f)
    existing_results = Results.model_validate(json_data)


    result, error_message = stt_service._run_inference(
        file_path=file_path_input,
        existing_results=existing_results
    )

    # for seg in result.segments:
    #     if seg.type == SegmentType.speech:
    #         logger.info(seg.text)

    # logger.debug(result.model_dump_json(indent=4))

    with open(f"{ROOT_DIR}/tests/{os.path.basename(file_path_input).replace(".wav", "")}_stt_result.json", "w", encoding="utf-8") as f:
        f.write(result.model_dump_json(indent=4))

