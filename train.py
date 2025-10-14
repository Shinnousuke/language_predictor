import os
import librosa
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
import joblib

# =========================
# Paths
# =========================
DATASET_PATH = r"C:\Users\Admin\Desktop\internship_projects\language_predictor\clened_dataset"
MODEL_PATH = "language_model.pkl"
ENCODER_PATH = "label_encoder.pkl"

# =========================
# Feature Extraction Function
# =========================
def extract_features(file_path, max_pad_len=174):
    """
    Extract MFCC features from an audio file.
    Output: Flattened vector of size 40*174 = 6960
    """
    try:
        audio, sr = librosa.load(file_path, sr=None)
        mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=40)

        # Pad or trim MFCCs to ensure fixed shape (40, 174)
        pad_width = max_pad_len - mfccs.shape[1]
        if pad_width > 0:
            mfccs = np.pad(mfccs, pad_width=((0, 0), (0, pad_width)), mode='constant')
        else:
            mfccs = mfccs[:, :max_pad_len]

        return mfccs.flatten()

    except Exception as e:
        print(f"⚠️ Error processing {file_path}: {e}")
        return None

# =========================
# Load Dataset
# =========================
def load_data():
    X, y = [], []
    for lang in os.listdir(DATASET_PATH):
        lang_path = os.path.join(DATASET_PATH, lang)
        if os.path.isdir(lang_path):
            print(f"📂 Loading language: {lang}")
            for file in os.listdir(lang_path):
                if file.lower().endswith(".wav"):
                    file_path = os.path.join(lang_path, file)
                    features = extract_features(file_path)
                    if features is not None:
                        X.append(features)
                        y.append(lang)

    print(f"✅ Total files loaded: {len(X)}")
    return np.array(X), np.array(y)

# =========================
# Train Model
# =========================
if __name__ == "__main__":
    print("🔍 Loading dataset...")
    X, y = load_data()

    print("🔤 Encoding labels...")
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)

    print("✂️ Splitting dataset...")
    X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded)

    print("🌲 Training RandomForest model...")
    model = RandomForestClassifier(n_estimators=200, random_state=42)
    model.fit(X_train, y_train)

    accuracy = model.score(X_test, y_test)
    print(f"✅ Model Accuracy: {accuracy * 100:.2f}%")

    print("💾 Saving model and label encoder...")
    joblib.dump(model, MODEL_PATH)
    joblib.dump(le, ENCODER_PATH)

    print("🎉 Training completed successfully!")
