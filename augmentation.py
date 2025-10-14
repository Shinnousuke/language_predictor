import os
import random
import librosa
import soundfile as sf
import numpy as np
import torchaudio
import torchaudio.transforms as T
import torch

# ========== CONFIG ==========
base_dir = r"C:\Users\Admin\Desktop\internship_projects\language_predictor\clened_dataset"
target_sr = 16000  # sample rate
# ============================

# -------- Step 1: Count samples per class --------
def get_class_counts(base_dir):
    """Count files per language class."""
    class_counts = {}
    for lang in os.listdir(base_dir):
        lang_path = os.path.join(base_dir, lang)
        if os.path.isdir(lang_path):
            count = len([f for f in os.listdir(lang_path) if f.endswith(".wav")])
            class_counts[lang] = count
    return class_counts

# -------- Step 2: Waveform augmentations --------
def augment_audio(y, sr):
    """Generate augmented versions of one audio sample."""
    augmented = []

    # Pitch shift
    n_steps = random.choice([-2, -1, 1, 2])
    augmented.append(librosa.effects.pitch_shift(y, sr=sr, n_steps=n_steps))

    # Speed change
    rate = random.choice([0.9, 1.1])
    try:
        augmented.append(librosa.effects.time_stretch(y, rate))
    except:
        pass  # skip very short clips

    # Add background noise
    noise = np.random.randn(len(y))
    augmented.append(y + 0.005 * noise)

    return augmented

# -------- Step 3: Balance dataset with augmentation --------
def balance_with_augmentation(base_dir):
    """Balance dataset by augmenting smaller classes."""
    counts = get_class_counts(base_dir)
    target_size = max(counts.values())
    print("Initial class counts:", counts)
    print("Target size:", target_size)

    for lang, count in counts.items():
        lang_path = os.path.join(base_dir, lang)
        files = [f for f in os.listdir(lang_path) if f.endswith(".wav")]

        print(f"\nProcessing {lang} ({count} files)...")

        i = 0
        while len(files) < target_size:
            file = random.choice(files)
            file_path = os.path.join(lang_path, file)

            # Load audio
            y, sr = librosa.load(file_path, sr=target_sr)

            # Augment
            augmented_audios = augment_audio(y, sr)

            for aug in augmented_audios:
                if len(files) >= target_size:
                    break
                new_name = f"aug_{i}_{file}"
                out_path = os.path.join(lang_path, new_name)
                sf.write(out_path, aug, sr)
                files.append(new_name)
                i += 1

        print(f"{lang} balanced to {len(files)} files ✅")

# -------- Step 4: SpecAugment (applied during training) --------
def get_augmented_spectrogram(file_path, sr=16000):
    """Load file -> MelSpectrogram -> Apply SpecAugment."""
    waveform, sample_rate = torchaudio.load(file_path)
    if sample_rate != sr:
        waveform = torchaudio.functional.resample(waveform, sample_rate, sr)

    # Convert to Mel-spectrogram
    mel_spec = T.MelSpectrogram(
        sample_rate=sr,
        n_mels=64
    )(waveform)

    # Apply SpecAugment (time & freq masking)
    spec_aug = T.FrequencyMasking(freq_mask_param=15)(mel_spec)
    spec_aug = T.TimeMasking(time_mask_param=35)(spec_aug)

    return spec_aug

# ====================== RUN ======================
if __name__ == "__main__":
    # Step 1–3: Balance dataset with waveform augmentations
    balance_with_augmentation(base_dir)

    # Example: apply SpecAugment to a sample file (training time)
    sample_file = None
    for lang in os.listdir(base_dir):
        lang_path = os.path.join(base_dir, lang)
        if os.path.isdir(lang_path):
            sample_file = os.path.join(lang_path, os.listdir(lang_path)[0])
            break

    if sample_file:
        spec_aug = get_augmented_spectrogram(sample_file)
        print(f"\nSpecAugment applied to {sample_file}, shape:", spec_aug.shape)
