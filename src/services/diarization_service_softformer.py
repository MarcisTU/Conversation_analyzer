import argparse
import asyncio
import json
import os
import sys
import time
import shutil
from typing import Tuple, List

import librosa
import torch
import numpy as np
from dotenv import load_dotenv
from loguru import logger
from nemo.collections.asr.models import SortformerEncLabelModel
from nemo.collections.asr.parts.mixins.diarization import DiarizeConfig
import nemo.collections.asr as nemo_asr
from tqdm import tqdm

from src.models.ai_speech import AiSADSegment

from src.modules.constants import ROOT_DIR, SRC_DIR
from sklearn import preprocessing

from src.models.enums import SegmentType
from src.models.results import Results
from src.models.results import ResultsSegment
from src.services.base_service import BaseService


class DiarizationService(BaseService):
    def __init__(self, args, batch_size=1):
        try:
            super().__init__()
            self.args = args
            self.batch_size = batch_size

            self.micro_pause_segment_merge_threshold = 1.0
            self.jre_podcast_intro_offset = 12.0   # for analyzing Joe Rogan Experience podcasts

            self.speaker_diarizer = SortformerEncLabelModel.from_pretrained("nvidia/diar_streaming_sortformer_4spk-v2.1")
            self.speaker_diarizer.eval()
            self.speaker_diarizer.sortformer_modules.chunk_len = 340
            self.speaker_diarizer.sortformer_modules.chunk_right_context = 40
            self.speaker_diarizer.sortformer_modules.fifo_len = 40
            self.speaker_diarizer.sortformer_modules.spkcache_update_period = 300

            logger.info(f"Using device: {self.args.device}")

            try:
                self.speaker_model = nemo_asr.models.EncDecSpeakerLabelModel.from_pretrained("nvidia/speakerverification_en_titanet_large")
            except Exception as exc:
                raise FileNotFoundError(f"Error loading speaker model: {exc}")

        except Exception as exc:
            logger.exception(exc)

    def normalize_embedding(self, np_embedding: np.ndarray) -> np.ndarray:
        np_result = np.array(np_embedding)

        is_squeeze = False
        if len(np_embedding.shape) < 2:
            is_squeeze = True
            np_result = np.expand_dims(np_result, axis=0)

        np_result = preprocessing.normalize(np_result)
        if is_squeeze:
            np_result = np_result.squeeze(axis=0)

        return np_result

    def merge_autoresponder_micro_pauses(self, segments_speech: List[AiSADSegment]) -> List[AiSADSegment]:
        segments_speech_processed = []
        skip_next_segment = False
        last_were_processed = True
        iter_count = 0
        while last_were_processed:
            iter_count += 1
            logger.debug(f"Iter: {iter_count} - Processing micro pauses between segments for autoresponder")
            last_were_processed = False
            for idx_segment in range(len(segments_speech) - 1):
                if skip_next_segment:
                    skip_next_segment = False

                    # check if last iteration and skip_next_segment is false; add last segment
                    if idx_segment == len(segments_speech) - 2:
                        idx_segment_next = idx_segment + 1
                        segment_next = segments_speech[idx_segment_next]
                        segments_speech_processed.append(segment_next)
                        break
                    else:
                        continue

                idx_segment_next = idx_segment + 1
                segment = segments_speech[idx_segment]
                segment_next = segments_speech[idx_segment_next]
                delta_sec = segment_next.start_sec - segment.end_sec
                if delta_sec < self.micro_pause_segment_merge_threshold:
                    segment.end_sec = segment_next.end_sec
                    skip_next_segment = True
                    last_were_processed = True

                segments_speech_processed.append(segment)

                if idx_segment == len(segments_speech) - 2 and not skip_next_segment:
                    segments_speech_processed.append(segment_next)

            if last_were_processed:  # need to process again to check if any fractured segments left
                segments_speech = segments_speech_processed
                segments_speech_processed = []
                skip_next_segment = False

        return segments_speech_processed

    def process_remove_short_segments(self, speech_activity_detection_regions):
        speech_activity_detection_regions = sorted(speech_activity_detection_regions, key=lambda x: x.start_sec)
        speech_activity_detection_regions_processed = []
        for idx, segment in enumerate(speech_activity_detection_regions):
            segment_length = segment.end_sec - segment.start_sec
            has_close_neighbor = False

            if idx > 0:
                prev_segment = speech_activity_detection_regions[idx - 1]
                if segment.start_sec - prev_segment.end_sec < 2.0:
                    has_close_neighbor = True

            if idx < len(speech_activity_detection_regions) - 1:
                next_segment = speech_activity_detection_regions[idx + 1]
                if next_segment.start_sec - segment.end_sec < 2.0:
                    has_close_neighbor = True

            if segment_length >= 1.0 or has_close_neighbor:
                speech_activity_detection_regions_processed.append(segment)
            else:
                logger.info(
                    f"Removing short isolated segment {segment.start_sec}-{segment.end_sec} sec (length {segment_length}s)")

        return speech_activity_detection_regions_processed

    async def inference(
        self,
        file_path: str,
        num_speakers=None
    ) -> Tuple[Results, str]:
        """
        Asynchronous wrapper for the diarization pipeline.
        This offloads the heavy CUDA/CPU computation to a separate OS thread,
        preventing the asyncio event loop from freezing.
        """
        logger.info(f"Starting async thread for diarization inference on: {file_path}")

        return await asyncio.to_thread(
            self._run_inference,
            file_path=file_path,
            num_speakers=num_speakers
        )

    def _run_inference(
        self,
        file_path: str,  # TODO in worker save the files from MinIO buckets to temp request file dir
        num_speakers = None
    ) -> Tuple[Results, str]:
        result = None
        error_message = None
        try:
            file_length_sec = librosa.get_duration(filename=file_path)

            audio, _ = librosa.load(file_path, sr=self.args.datasource_samplerate)

            # skip first 12sec intro for JRE podcast
            audio = audio[int(self.args.datasource_samplerate * self.jre_podcast_intro_offset):]

            predicted_segments = self.speaker_diarizer.diarize(
                audio=audio,
                batch_size=self.batch_size,
                override_config=DiarizeConfig(
                    batch_size=2,
                    max_num_of_spks=3,
                    sample_rate=self.args.datasource_samplerate
                )
            )
            logger.debug(predicted_segments)

            speech_activity_detection_regions = []
            for segment in predicted_segments[0]:
                seg_parts = segment.split(" ")
                start_sec = float(seg_parts[0])
                end_sec = float(seg_parts[1])
                speaker_id = int(seg_parts[2].replace("speaker_", ""))
                speech_activity_detection_regions.append(AiSADSegment(
                    start_sec=start_sec,
                    end_sec=end_sec,
                    speaker_id=speaker_id
                ))

            speech_activity_detection_regions = self.process_remove_short_segments(speech_activity_detection_regions)

            # Extract embeddings for conversation part
            speaker_segments = {}
            y, sr = librosa.load(file_path, sr=self.args.datasource_samplerate)
            for idx_reg, sad_region in enumerate(speech_activity_detection_regions):
                if sad_region.speaker_id not in speaker_segments:
                    speaker_segments[sad_region.speaker_id] = []

                wav_segment = y[int(sad_region.start_sec * self.args.datasource_samplerate):int(sad_region.end_sec * self.args.datasource_samplerate)]
                speaker_segments[sad_region.speaker_id].append(wav_segment)

            np_conversation_embeddings = {}
            for speaker_id, segments in speaker_segments.items():
                all_seg_embeddings = []
                for seg in tqdm(segments, desc="Extracting embeddings from segments"):
                    input_embedding, _ = self.speaker_model.infer_segment(
                        segment=seg
                    )
                    all_seg_embeddings.append(input_embedding)

                # concat list of embs and reduce from (n, 192) -> (192)
                input_embedding = torch.cat(all_seg_embeddings, dim=0)
                input_embedding = torch.median(input_embedding, dim=0)[0]

                if len(input_embedding.shape) == 1:
                    input_embedding = input_embedding.unsqueeze(0)

                np_conversation_embeddings[speaker_id] = input_embedding.cpu().numpy()

            np_conversation_embeddings_processed = {}
            for speaker_id, embeddings in np_conversation_embeddings.items():
                embeddings = self.normalize_embedding(embeddings)
                embeddings = np.squeeze(embeddings)
                np_conversation_embeddings_processed[speaker_id] = embeddings

            np_conversation_embeddings_processed_np = np.array(list(np_conversation_embeddings_processed.values()))
            logger.debug(f'np_conversation_embeddings_processed_np: {np_conversation_embeddings_processed_np.shape}')

            ### check if np_conversation_embeddings_processed_np is empty and if it is create empty speech segment in results and skip rest
            if len(np_conversation_embeddings_processed_np) == 0:
                logger.warning(f"No conversation embeddings found. Setting to empty noise segment.")
                segment_noise = ResultsSegment()
                segment_noise.type = SegmentType.noise
                segment_noise.start_time = 0.0
                segment_noise.end_time = file_length_sec

                result = Results()
                result.length_sec = file_length_sec
                result.segments = [segment_noise]

            else:
                speaker_id_map = {}
                for idx, speaker_id in enumerate(np_conversation_embeddings.keys()):
                    speaker_id_map[speaker_id] = -1 * (idx + 1)

                segments_speech = []
                for idx_reg, sad_region in enumerate(speech_activity_detection_regions):
                    user_id = speaker_id_map[sad_region.speaker_id]

                    segment = ResultsSegment()
                    segment.type = SegmentType.speech
                    segment.user_id = user_id
                    segment.start_time = round(sad_region.start_sec + self.jre_podcast_intro_offset, 2)
                    segment.end_time = round(sad_region.end_sec + self.jre_podcast_intro_offset, 2)
                    segments_speech.append(segment)

                segments_speech = sorted(segments_speech, key=lambda x: x.start_time)

                # remove segments that are smaller or equal than 0.2sec; too fractured for any information to be spoken
                segments_speech = [segment for segment in segments_speech if segment.end_time - segment.start_time > 0.25]

                # remove micro pauses between segments that are less than 0.3 sec long and merge segments if their user_id is the same
                segments_speech_processed = []
                skip_next_segment = False
                last_were_processed = True
                iter_count = 0
                while last_were_processed:
                    if len(segments_speech) <= 1:  # nothing to process for merging
                        segments_speech_processed = segments_speech
                        logger.debug("Only one segment left, no need to process micro pauses.")
                        break

                    iter_count += 1
                    logger.debug(f"Iter: {iter_count} - Processing micro pauses between segments")
                    last_were_processed = False
                    for idx_segment in range(len(segments_speech) - 1):
                        if skip_next_segment:
                            skip_next_segment = False

                            # check if last iteration and skip_next_segment is false; add last segment
                            if idx_segment == len(segments_speech) - 2:
                                idx_segment_next = idx_segment + 1
                                segment_next = segments_speech[idx_segment_next]
                                segments_speech_processed.append(segment_next)
                                break
                            else:
                                continue

                        idx_segment_next = idx_segment + 1
                        segment = segments_speech[idx_segment]
                        segment_next = segments_speech[idx_segment_next]
                        delta_sec = segment_next.start_time - segment.end_time
                        if delta_sec < self.micro_pause_segment_merge_threshold and segment.user_id == segment_next.user_id and segment_next.start_time >= segment.end_time:
                            logger.debug(f"Removing micro pause between segments {segment.end_time} - {segment_next.start_time} sec")
                            segment.end_time = segment_next.end_time
                            skip_next_segment = True
                            last_were_processed = True

                        segments_speech_processed.append(segment)

                        if idx_segment == len(segments_speech) - 2 and not skip_next_segment:
                            segments_speech_processed.append(segment_next)

                    if last_were_processed:  # need to process again to check if any fractured segments left
                        segments_speech = segments_speech_processed
                        segments_speech_processed = []
                        skip_next_segment = False

                segments_speech = segments_speech_processed
                logger.debug(f"Processed {iter_count} times to remove micro pauses between segments")

                # Add noise segments between speech segments but keep also speech segments
                segments_speech_all = []
                last_sec = 0
                for segment_speech in segments_speech:
                    len_sec = segment_speech.start_time - last_sec
                    if len_sec > 0:
                        segment_noise = ResultsSegment()
                        segment_noise.type = SegmentType.noise
                        segment_noise.start_time = last_sec
                        segment_noise.end_time = segment_speech.start_time
                        segments_speech_all.append(segment_noise)
                    last_sec = segment_speech.end_time

                    if segment_speech.type == SegmentType.speech:
                        segments_speech_all.append(segment_speech)

                # if segments_speech_all empty add one big noise type segment for the whole file duration
                if len(segments_speech_all) == 0:
                    segment_noise = ResultsSegment()
                    segment_noise.type = SegmentType.noise
                    segment_noise.start_time = 0.0
                    segment_noise.end_time = file_length_sec
                    segments_speech_all.append(segment_noise)

                result = Results()
                result.length_sec = file_length_sec
                result.segments = segments_speech_all

        except Exception as exc:
            logger.exception(exc)
            if error_message is None:
                error_message = "(type=%r, value=%r, traceback=%r)" % sys.exc_info()

        return result, error_message


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-device', default='cuda', type=str)
    parser.add_argument('-datasource_samplerate', default=16000, type=int)
    args, _ = parser.parse_known_args()

    load_dotenv(f"{ROOT_DIR}/.env")

    controller_diarization = DiarizationService(args)
    target_sample_rate = 16000

    # file_path_input = "../voiceid_files/unprocessed/714/voice_images-male-31-lv-1516-714-0-867bec91-04d9-4f39-b551-58b546133f60.wav"
    # np_voice_embeddings, quality = controller_diarization.get_voiceid_file_embeddings(input_audio_path=file_path_input, num_speakers=1)
    # logger.info(f'np_voice_embeddings shape: {np_voice_embeddings[0].shape}')
    # logger.info(f'quality: {quality}')

    # file_path_input = f"{ROOT_DIR}/tests/KT_file_1_test_mono.wav"
    file_path_input = f"{ROOT_DIR}/tests/JRE_Chase_Hughes_16k_mono.wav"

    # AudioUtils.get_wav_info(file_path_input)
    # AudioUtils.convert_stereo_to_mono(file_path=file_path_input, output_path=f"{ROOT_DIR}/tests/KT_file_1_test_mono.wav")
    #
    # exit()

    # reference_audio_files = glob.glob("../voiceid_files/unprocessed/714/voice_*.wav")
    # reference_audio_files = glob.glob("../voiceid_files/unprocessed/120/voice_*.wav")
    # reference_embeddings = []
    # for ref_audio in reference_audio_files:
    #     embedding, quality = controller_diarization.get_voiceid_file_embeddings(input_audio_path=ref_audio)
    #     reference_embeddings.append(embedding)

    # concat embedings
    # reference_embeddings = np.concatenate(reference_embeddings, axis=0)
    #
    # voiceid_conversation_input = VoiceIdConversationInput(num_speakers=2, segment_min_sec=1.0)
    # member = VoiceIdConversationMember()
    # member.internal_user_id = 120
    # member.client_id = 300
    # member.client_user_id = 2
    # member.np_voiceprints = reference_embeddings
    # voiceid_conversation_input.members.append(member)

    result, error_message = controller_diarization._run_inference(
        file_path=file_path_input,
        # voiceid_conversation_input,
        # task_client_user_id=member.client_id,
        # is_process_autoresponder=False
    )

    print(result.model_dump_json(indent=4))

    with open(f"{ROOT_DIR}/tests/{os.path.basename(file_path_input).replace(".wav", "")}_diarize_result.json", "w", encoding="utf-8") as f:
        f.write(result.model_dump_json(indent=4))

