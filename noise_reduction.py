import os
import librosa
import noisereduce as nr
import soundfile as sf

# === SETTINGS ===
input_folder = r"C:\Users\Admin\Desktop\internship_projects\language_predictor\dataset"
output_folder = r"C:\Users\Admin\Desktop\internship_projects\language_predictor\clened_dataset"

# Walk through all subfolders
for root, dirs, files in os.walk(input_folder):
    for file in files:
        if file.lower().endswith(".wav"):
            input_path = os.path.join(root, file)

            # Make relative path (to keep subfolder structure)
            rel_path = os.path.relpath(root, input_folder)
            output_subfolder = os.path.join(output_folder, rel_path)

            # Create subfolder if it doesn't exist
            os.makedirs(output_subfolder, exist_ok=True)

            # Output file path
            output_path = os.path.join(output_subfolder, file)

            print(f"Processing: {input_path}")

            # Load audio
            y, sr = librosa.load(input_path, sr=None)

            # Apply noise reduction
            reduced_noise = nr.reduce_noise(y=y, sr=sr, prop_decrease=1.0)

            # Save cleaned file
            sf.write(output_path, reduced_noise, sr)

            print(f"Saved cleaned file: {output_path}")

print("\n✅ Noise removal completed for all files and saved in same folder structure!")
