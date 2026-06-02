import os

import soundfile as sf
from loguru import logger


class AudioUtils:
    @staticmethod
    def get_wav_info(file_path):
        info = sf.info(file_path)

        logger.info("--- Audio File Data ---")
        logger.info(f"File Path:     {file_path}")
        logger.info(f"Channels:      {info.channels} ({'Mono / Single Channel' if info.channels == 1 else 'Stereo / Multi-Channel'})")
        logger.info(f"Sample Rate:   {info.samplerate} Hz")
        logger.info(f"Duration:      {info.duration:.2f} seconds")
        logger.info(f"Total Frames:  {info.frames}")
        logger.info(f"Subtype/Bits:  {info.subtype} (e.g., PCM_16, FLOAT)")

    @staticmethod
    def convert_stereo_to_mono(file_path, output_path=None):
        """
        Checks if a WAV file is mono. If it's stereo/multi-channel,
        converts it to mono by averaging the channels and saves it.

        :param file_path: Path to the input WAV file.
        :param output_path: Path to save the mono WAV file. If None, overwrites the input file.

        :return: Path to the mono audio file.
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Audio file not found at: {file_path}")


        info = sf.info(file_path)
        file_path_result = file_path
        if info.channels == 1:
            logger.info(f"✅ Already Mono: '{file_path}' has 1 channel. No conversion needed.")
        else:
            logger.info(f"⚠️ Stereo Detected: '{file_path}' has {info.channels} channels. Converting to mono...")
            data, sample_rate = sf.read(file_path)

            # This blends Left and Right channels together perfectly without clipping
            mono_data = data.mean(axis=1)

            target_path = output_path if output_path else file_path

            sf.write(target_path, mono_data, sample_rate, subtype=info.subtype)
            logger.info(f"🚀 Success: Mono file saved to '{target_path}'")

            file_path_result = target_path

        return file_path_result
