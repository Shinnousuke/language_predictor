import streamlit as st
import numpy as np
import os
import joblib
import librosa
import tempfile
from pydub import AudioSegment
from scipy.special import softmax

# =========================
# App Configuration
# =========================
st.set_page_config(page_title="Language Detector ", layout="centered")
st.title("🌍 Voice Language Detector ")
st.markdown("Upload an audio file to detect the **spoken language** using your trained ML model.")

# =========================
# Constants
# =========================
SAMPLE_RATE = 16000
TARGET_FEATURES = 6960

# =========================
# Load Model & Label Encoder
# =========================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

try:
    model = joblib.load(os.path.join(BASE_DIR, "language_model.pkl"))
    label_encoder = joblib.load(os.path.join(BASE_DIR, "label_encoder.pkl"))
except Exception:
    st.error("❌ Model or label encoder not found. Please make sure both files exist in the app directory.")
    st.stop()

# =========================
# Feature Extraction (MFCC)
# =========================
def extract_features(file_path, max_pad_len=174):
    """Extract 40 MFCCs × 174 frames = 6960 features."""
    try:
        audio, sr = librosa.load(file_path, sr=None)
        if audio.ndim > 1:
            audio = np.mean(audio, axis=1)
        if sr != SAMPLE_RATE:
            audio = librosa.resample(audio, orig_sr=sr, target_sr=SAMPLE_RATE)
            sr = SAMPLE_RATE

        mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=40)
        pad_width = max_pad_len - mfccs.shape[1]
        if pad_width > 0:
            mfccs = np.pad(mfccs, pad_width=((0, 0), (0, pad_width)), mode="constant")
        else:
            mfccs = mfccs[:, :max_pad_len]

        return mfccs.flatten().reshape(1, -1)
    except Exception as e:
        st.error(f"Error extracting features: {e}")
        return np.zeros((1, TARGET_FEATURES))

# =========================
# Prediction with Softmax
# =========================
def predict_language_with_confidence(features):
    """Predict language and show top 3 probabilities using Softmax."""
    try:
        if hasattr(model, "predict_proba"):
            probs = model.predict_proba(features)[0]
        elif hasattr(model, "decision_function"):
            scores = model.decision_function(features)
            probs = softmax(scores, axis=1)[0]
        else:
            pred = model.predict(features)[0]
            probs = np.zeros(len(label_encoder.classes_))
            probs[pred] = 1.0

        top_indices = np.argsort(probs)[::-1][:3]
        top_langs = label_encoder.inverse_transform(top_indices)
        top_probs = probs[top_indices] * 100

        detected_language = top_langs[0]
        return detected_language, list(zip(top_langs, top_probs))
    except Exception as e:
        st.error(f"Prediction error: {e}")
        return None, []

# =========================
# Upload Audio Section
# =========================
st.subheader("📤 Upload Audio File and Detect Language")
uploaded_file = st.file_uploader("Upload an audio file (mp3, wav, m4a)", type=["mp3", "wav", "m4a"])

if uploaded_file is not None:
    temp_dir = "temp_audio"
    os.makedirs(temp_dir, exist_ok=True)
    file_path = os.path.join(temp_dir, uploaded_file.name)

    with open(file_path, "wb") as f:
        f.write(uploaded_file.read())

    st.audio(file_path)

    if st.button("🔍 Detect Language"):
        try:
            if file_path.endswith((".mp3", ".m4a")):
                wav_path = file_path.rsplit(".", 1)[0] + ".wav"
                sound = AudioSegment.from_file(file_path)
                sound.export(wav_path, format="wav")
            else:
                wav_path = file_path

            features = extract_features(wav_path)
            detected_language, top_preds = predict_language_with_confidence(features)

            if detected_language:
                st.success(f"🗣️ **Detected Language:** {detected_language}")
                st.markdown("### 🔢 Prediction Confidence:")
                for lang, prob in top_preds:
                    st.write(f"- {lang}: **{prob:.2f}%**")

        except Exception as e:
            st.error("❌ Could not detect language from uploaded audio.")
            st.exception(e)
        finally:
            try:
                os.remove(file_path)
                if os.path.exists(wav_path) and wav_path != file_path:
                    os.remove(wav_path)
            except:
                pass

# =========================
# Footer
# =========================
st.markdown("---")
st.caption("👩‍💻 Developed by Aditi | Powered by Streamlit & Custom ML Model with Softmax Confidence")
