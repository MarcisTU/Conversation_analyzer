import argparse
import asyncio
import json
import sys
from typing import Tuple, Optional
import itertools
from collections import Counter

import librosa
import numpy as np
from dotenv import load_dotenv
from loguru import logger
from funasr import AutoModel

from src.modules.constants import ROOT_DIR, SRC_DIR
from src.models.enums import SegmentType, EmotionLabel
from src.models.results import Results, ResultEmotion
from src.utils.audio_utils import AudioUtils


class AudioEmotionService:
    def __init__(self, args, batch_size=2):
        try:
            super().__init__()
            self.args = args
            self.batch_size = batch_size
            self.long_segment_threshold = 5.0
            self.segment_padding = 1.0  # padding around audio chunk segments

            try:
                model_id = "iic/emotion2vec_plus_large"
                self.emo_model = AutoModel(
                    model=model_id,
                    hub="hf"
                )
                logger.info(f"Audio Emotion model initialized!")
            except Exception as exc:
                raise FileNotFoundError(f"Error loading audio emotion recognition model: {exc}")

        except Exception as exc:
            logger.exception(exc)

    def split_segment(self, segment):
        """Split long segment into chunks of at self.long_segment_threshold seconds, and each padded."""
        duration = segment.end_time - segment.start_time
        if duration <= self.long_segment_threshold:
            return [(
                max(0.0, segment.start_time - self.segment_padding),
                segment.end_time + self.segment_padding
            )]

        chunks = []
        cursor = segment.start_time
        while cursor < segment.end_time:
            chunk_end = min(cursor + self.long_segment_threshold, segment.end_time)
            chunks.append((
                max(0.0, cursor - self.segment_padding),
                chunk_end + self.segment_padding
            ))
            cursor = chunk_end
        return chunks

    def aggregate_chunk_emotions(self, chunk_emotions: list[ResultEmotion]) -> ResultEmotion:
        """Aggregate multiple chunk emotions into one by averaging confidence per label."""

        if len(chunk_emotions) == 1:
            out_result = chunk_emotions[0]
        else:
            label_scores: dict[EmotionLabel, list[float]] = {}
            for emo in chunk_emotions:
                label_scores.setdefault(emo.label, []).append(emo.confidence)

            avg_scores = {label: sum(scores) / len(scores) for label, scores in label_scores.items()}
            best_label = max(avg_scores, key=avg_scores.get)

            out_result = ResultEmotion(label=best_label, confidence=avg_scores[best_label])

        return out_result

    def parse_emotion_result(self, res) -> ResultEmotion:
        labels = res['labels']
        scores = res['scores']
        max_idx = scores.index(max(scores))
        raw_label = labels[max_idx]
        clean_label = raw_label.split('/')[-1] if '/' in raw_label else raw_label

        if clean_label == "<unk>":
            clean_label = EmotionLabel.unknown
        else:
            clean_label = EmotionLabel(clean_label)

        return ResultEmotion(label=clean_label, confidence=scores[max_idx])

    async def inference(
        self,
        file_path: str,
        file_path_denoised: str,
        existing_results: Results
    ) -> Tuple[Results, str]:
        """
        Asynchronous wrapper for the emotions pipeline.
        This offloads the heavy CUDA/CPU computation to a separate OS thread,
        preventing the asyncio event loop from freezing.
        """
        logger.info(f"Starting async thread for emotions inference on: {file_path}")

        return await asyncio.to_thread(
            self._run_inference,
            file_path=file_path,
            file_path_denoised=file_path_denoised,
            existing_results=existing_results
        )

    def _run_inference(
        self,
        file_path: str,
        file_path_denoised: Optional[str],
        existing_results: Results
    ) -> Tuple[Results, str]:
        error_message = None
        try:
            if file_path_denoised is not None and file_path_denoised != file_path:
                file_path = file_path_denoised

            y, sr = librosa.load(file_path, sr=self.args.datasource_samplerate)

            all_segments_speech = []
            all_segments_speech_idxs = []
            for idx_s, segment in enumerate(existing_results.segments):
                if segment.type == SegmentType.speech:
                    all_segments_speech.append(segment)
                    all_segments_speech_idxs.append(idx_s)

            # Build flat list of (seg_idx, chunk_audio) pairs, tracking how many chunks per segment are used
            seg_chunk_map: list[int] = []  # maps flat chunk index → segment index in all_segments_speech_idxs
            flat_chunks: list[np.ndarray] = []
            for seg_list_idx, segment in enumerate(all_segments_speech):
                chunks = self.split_segment(segment)
                for chunk_start, chunk_end in chunks:
                    flat_chunks.append(AudioUtils.audio_slice(y, chunk_start, chunk_end, self.args.datasource_samplerate))
                    seg_chunk_map.append(seg_list_idx)

            # Process all chunks in batches
            flat_emotions: list[ResultEmotion] = []
            for batch_chunks in itertools.batched(flat_chunks, n=self.batch_size):
                try:
                    emo_results = self.emo_model.generate(list(batch_chunks), granularity="utterance", extract_embedding=False)

                    for res in emo_results:
                        flat_emotions.append(self.parse_emotion_result(res))
                except Exception as exc:
                    logger.error(f"Failed emotion detection for batch: {exc}")
                    for _ in batch_chunks:
                        flat_emotions.append(ResultEmotion(label=EmotionLabel.unknown, confidence=0.0))

            # Aggregate chunks back to segments
            chunks_per_segment: dict[int, list[ResultEmotion]] = {}
            for chunk_idx, seg_list_idx in enumerate(seg_chunk_map):
                chunks_per_segment.setdefault(seg_list_idx, []).append(flat_emotions[chunk_idx])

            for seg_list_idx, chunk_emotions in chunks_per_segment.items():
                seg_idx = all_segments_speech_idxs[seg_list_idx]
                existing_results.segments[seg_idx].emotions.append(self.aggregate_chunk_emotions(chunk_emotions))

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

    audio_emotion_service = AudioEmotionService(args)

    file_path_input = f"{ROOT_DIR}/tests/KT_file_1_test_mono.wav"

    with open(f"{ROOT_DIR}/tests/diarize_result.json", "r", encoding="utf-8") as f:
        json_data = json.load(f)
    existing_results = Results.model_validate(json_data)


    results, error_message = audio_emotion_service._run_inference(
        file_path=file_path_input,
        file_path_denoised=file_path_input,
        existing_results=existing_results
    )

    emotion_counts = Counter(
        emotion.label
        for segment in results.segments
        if segment.type == "speech"
        for emotion in segment.emotions
    )

    print(results.model_dump_json(indent=4))

    with open(f"{ROOT_DIR}/tests/audio_emotions_result.json", "w", encoding="utf-8") as f:
        f.write(results.model_dump_json(indent=4))

