from pydub import AudioSegment


input_file = "./JRE_Chase_Hughes.wav"
output_file = "./JRE_Chase_Hughes_16k_mono.wav"

# Load WAV file
audio = AudioSegment.from_wav(input_file)

# Convert to mono and 16 kHz
audio = audio.set_channels(1)
audio = audio.set_frame_rate(16000)

# Export converted file
audio.export(output_file, format="wav")

print(f"Saved converted file to: {output_file}")