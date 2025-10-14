import os
from pydub import AudioSegment
from pydub.utils import which

# Make sure ffmpeg is detected
AudioSegment.converter = which("ffmpeg")

DATASET_PATH = "cv-corpus-20.0-2024-12-06"
SAMPLE_RATE = 16000

for lang in os.listdir(DATASET_PATH):
    lang_path = os.path.join(DATASET_PATH, lang)
    if not os.path.isdir(lang_path):
        continue

    # Walk through subfolders (like clips)
    for root, _, files in os.walk(lang_path):
        for file in files:
            if file.endswith(".mp3"):
                mp3_path = os.path.join(root, file)
                wav_path = os.path.splitext(mp3_path)[0] + ".wav"

                try:
                    audio = AudioSegment.from_mp3(mp3_path)
                    audio = audio.set_frame_rate(SAMPLE_RATE).set_channels(1)
                    audio.export(wav_path, format="wav")
                    print(f"Converted: {mp3_path} -> {wav_path}")
                except Exception as e:
                    print(f"Error converting {mp3_path}: {e}")
