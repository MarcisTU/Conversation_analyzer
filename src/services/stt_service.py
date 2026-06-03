import argparse
import asyncio
import json
import sys
from typing import Tuple, List

import librosa
import torch
import numpy as np
from dotenv import load_dotenv
from loguru import logger
from funasr import AutoModel

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
                pass
            except Exception as exc:
                raise FileNotFoundError(f"Error loading audio emotion recognition model: {exc}")

        except Exception as exc:
            logger.exception(exc)

    async def inference(
        self,
        file_path: str,
        file_path_denoised: str,
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
            file_path_denoised=file_path_denoised,
            existing_results=existing_results
        )

    def _run_inference(
        self,
        file_path: str,
        file_path_denoised: str,
        existing_results: Results
    ) -> Tuple[Results, str]:
        error_message = None
        try:
            pass

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

    gpus = []
    def get_gpu_usage():
        r_mb = torch.cuda.memory_reserved(0) / 1_000 / 1_000
        a_mb = torch.cuda.memory_allocated(0) / 1_000 / 1_000
        return {'reserved': r_mb, 'used': {a_mb}}

    stt_service = SttService(args)

    file_path_input = f"{ROOT_DIR}/tests/KT_file_1_test_mono.wav"

    with open("diarize_result.json", "r", encoding="utf-8") as f:
        json_data = json.load(f)
    existing_results = Results.model_validate(json_data)


    result, error_message = stt_service._run_inference(
        file_path=file_path_input,
        file_path_denoised=file_path_input,
        existing_results=existing_results
    )

    print(result.model_dump_json(indent=4))

    with open(f"{ROOT_DIR}/tests/stt_result.json", "w", encoding="utf-8") as f:
        f.write(result.model_dump_json(indent=4))

