import argparse
import asyncio
import json
import os
import sys
from pathlib import Path
from typing import Tuple, List

import librosa
import torch
import numpy as np
import torchaudio
from dotenv import load_dotenv
from loguru import logger
from funasr import AutoModel
from transformers import AutoProcessor, AutoModelForSpeechSeq2Seq

from src.modules.constants import ROOT_DIR, SRC_DIR
from src.models.enums import SegmentType
from src.models.results import Results
from src.models.results import ResultsSegment
from src.utils.audio_utils import AudioUtils


class SttService:
    def __init__(self, args, batch_size=2):
        try:
            super().__init__()
            self.args = args
            self.batch_size = batch_size

            try:
                self.device = "cuda" if torch.cuda.is_available() else "cpu"

                model_name = "ibm-granite/granite-speech-4.1-2b"
                self.processor = AutoProcessor.from_pretrained(model_name)
                self.tokenizer = self.processor.tokenizer
                self.model = AutoModelForSpeechSeq2Seq.from_pretrained(
                    model_name, device_map=self.device, torch_dtype=torch.bfloat16
                )

                self.max_new_tokens = 200
                self.num_beams = 1

            except Exception as exc:
                raise FileNotFoundError(f"Error loading audio emotion recognition model: {exc}")

        except Exception as exc:
            logger.exception(exc)

    def _find_longest_common_sequence(sequences, tokenizer):
        # TODO  Use a faster algorithm this can probably be done in O(n)
        # using suffix array.
        # It might be tedious to do because of fault tolerance.
        # We actually have a really good property which is that the total sequence
        # MUST be those subsequences in order.
        # Also the algorithm should be more tolerant to errors.
        sequence = [tok_id for tok_id in sequences[0][0].tolist() if tok_id not in tokenizer.all_special_ids]
        for new_seq in sequences[1:]:
            new_sequence = [tok_id for tok_id in new_seq[0].tolist() if tok_id not in tokenizer.all_special_ids]

            index = 0
            max_ = 0.0
            for i in range(1, len(new_sequence) + 1):
                # epsilon to favor long perfect matches
                eps = i / 10000.0
                matches = np.sum(np.array(sequence[-i:]) == np.array(new_sequence[:i]))
                matching = matches / i + eps
                if matches > 1 and matching > max_:
                    index = i
                    max_ = matching
            sequence.extend(new_sequence[index:])
        return np.array(sequence)

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
            # TODO implement segment chunking and processing. Use _find_longest_common_sequence [https://github.com/huggingface/transformers/blob/main/src/transformers/pipelines/automatic_speech_recognition.py]

            # Load audio
            # wav, sr = torchaudio.load(file_path, normalize=True)
            wav, sr = librosa.load(file_path, sr=16000)
            logger.debug(len(wav))
            logger.debug(wav.shape)
            # assert wav.shape[0] == 1 and sr == 16000  # mono, 16kHz

            wav = wav[int(28.0 * 16000):int(31.0 * 16000)]

            # Create text prompt
            user_prompt = "<|audio|>transcribe the speech with proper punctuation and capitalization."
            chat = [
                {"role": "user", "content": user_prompt},
            ]
            prompt = self.tokenizer.apply_chat_template(chat, tokenize=False, add_generation_prompt=True)

            # Run the processor + model
            model_inputs = self.processor(prompt, wav, device=self.device, return_tensors="pt").to(self.device)
            model_outputs = self.model.generate(
                **model_inputs, max_new_tokens=self.max_new_tokens, do_sample=False, num_beams=self.num_beams
            )

            # Transformers includes the input IDs in the response
            num_input_tokens = model_inputs["input_ids"].shape[-1]
            new_tokens = model_outputs[0, num_input_tokens:].unsqueeze(0)
            output_text = self.tokenizer.batch_decode(
                new_tokens, add_special_tokens=False, skip_special_tokens=True
            )
            logger.info(f"STT output = {output_text[0]}")

            # TODO edit existing_results here

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

    gpus = []
    def get_gpu_usage():
        r_mb = torch.cuda.memory_reserved(0) / 1_000 / 1_000
        a_mb = torch.cuda.memory_allocated(0) / 1_000 / 1_000
        return {'reserved': r_mb, 'used': {a_mb}}

    stt_service = SttService(args)

    file_path_input = f"{ROOT_DIR}/tests/KT_file_1_test_mono.wav"

    with open(f"{ROOT_DIR}/tests/diarize_result.json", "r", encoding="utf-8") as f:
        json_data = json.load(f)
    existing_results = Results.model_validate(json_data)


    result, error_message = stt_service._run_inference(
        file_path=file_path_input,
        existing_results=existing_results
    )

    # logger.debug(result.model_dump_json(indent=4))

    with open(f"{ROOT_DIR}/tests/stt_result.json", "w", encoding="utf-8") as f:
        f.write(result.model_dump_json(indent=4))

