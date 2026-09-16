import json
import threading
from pathlib import Path

from loguru import logger
import requests
from flask import Flask, request
from src.models.enums import FeatureName


API_URL = "http://localhost:8083/api/v1/tasks/submit"
CALLBACK_PORT = 8765
AUDIO_FILE = Path(__file__).parent / "KT_file_1_test_mono.wav"

app = Flask(__name__)
callback_received = threading.Event()
callback_result = None


@app.post("/callback")
def callback():
    global callback_result

    callback_result = request.get_json()

    logger.info("=" * 60)
    logger.info("Callback received!")
    logger.info("=" * 60)
    logger.info(json.dumps(callback_result, indent=4, ensure_ascii=False))

    callback_received.set()

    return {"status": "received"}, 200


def run_callback_server():
    app.run(
        host="0.0.0.0",
        port=CALLBACK_PORT,
        debug=False,
        use_reloader=False,
    )


def main():
    if not AUDIO_FILE.exists():
        raise FileNotFoundError(AUDIO_FILE)

    # Start callback server in background
    threading.Thread(
        target=run_callback_server,
        daemon=True,
    ).start()

    callback_url = (
        f"http://host.docker.internal:{CALLBACK_PORT}/callback"
    )

    features = [
        FeatureName.diarization.value,
        # FeatureName.emotion.value,
        # FeatureName.stt.value,
    ]

    params = [
        ("callback_url", callback_url),
        *[("features_to_process", feature) for feature in features],
    ]

    logger.info("Submitting task...")
    logger.info(f"Audio: {AUDIO_FILE}")
    logger.info(f"Callback: {callback_url}")

    with AUDIO_FILE.open("rb") as audio:
        response = requests.post(
            API_URL,
            params=params,
            files={
                "file": (
                    AUDIO_FILE.name,
                    audio,
                    "audio/wav",
                )
            },
        )

    response.raise_for_status()

    task = response.json()

    logger.info("Task submitted:")
    logger.info(json.dumps(task, indent=2))

    logger.info("Waiting for callback...")

    callback_received.wait()

    logger.info("Done.")


if __name__ == "__main__":
    main()
