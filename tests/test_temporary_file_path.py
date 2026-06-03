import time
import tempfile
from pathlib import Path
import librosa

from src.modules.constants import ROOT_DIR


def process_audio_file(file_path: str):
    print(f"Loading file from: {file_path}")

    y, sr = librosa.load(file_path, sr=None)

    print(f"Audio loaded successfully!")
    print(f"Sample Rate: {sr} Hz")
    print(f"Audio Data Shape: {y.shape}")


TEST_WAV_PATH = Path(f"{ROOT_DIR}/tests/KT_file_1_test_mono.wav")
LOCAL_TMP_DIR = Path(f"{ROOT_DIR}/tests/.tmp")

LOCAL_TMP_DIR.mkdir(parents=True, exist_ok=True)

try:
    if not TEST_WAV_PATH.exists():
        raise FileNotFoundError(f"Please put a test file at: {TEST_WAV_PATH.resolve()}")

    with open(TEST_WAV_PATH, "rb") as local_file:
        mock_minio_bytes = local_file.read()

    with tempfile.NamedTemporaryFile(
        suffix=".wav",
        dir=LOCAL_TMP_DIR,
        delete=True
    ) as temp_wav:
        temp_wav.write(mock_minio_bytes)
        temp_wav.flush()
        temp_wav.seek(0)

        print(f"Temporary file created at: {temp_wav.name}")
        # time.sleep(15)

        process_audio_file(temp_wav.name)

        print("Successfully simulated downstream function execution!")

except Exception as e:
    print(f"An error occurred during testing: {e}")