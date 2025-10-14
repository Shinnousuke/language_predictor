import streamlit as st
import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import welch

st.set_page_config(page_title="Audio Noise Visualizer", layout="wide")

st.title("🎵 Audio Noise Visualizer")

# Upload audio file
uploaded_file = st.file_uploader("Upload an audio file", type=["wav", "mp3", "ogg", "flac"])

if uploaded_file is not None:
    # Load audio
    y, sr = librosa.load(uploaded_file, sr=None)

    st.subheader("1️⃣ Waveform Plot")
    fig, ax = plt.subplots(figsize=(10, 3))
    librosa.display.waveshow(y, sr=sr, ax=ax)
    ax.set_title("Waveform")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Amplitude")
    st.pyplot(fig)

    st.subheader("2️⃣ Spectrogram (Frequency vs Time)")
    D = librosa.stft(y)
    S_db = librosa.amplitude_to_db(np.abs(D), ref=np.max)

    fig, ax = plt.subplots(figsize=(10, 5))
    img = librosa.display.specshow(S_db, sr=sr, x_axis='time', y_axis='hz', ax=ax)
    fig.colorbar(img, ax=ax, format="%+2.0f dB")
    ax.set_title("Spectrogram (Noise visible as spread)")
    st.pyplot(fig)

    st.subheader("3️⃣ Power Spectral Density (PSD)")
    f, Pxx = welch(y, sr, nperseg=1024)

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.semilogy(f, Pxx)
    ax.set_title("Power Spectral Density")
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("Power/Frequency (dB/Hz)")
    st.pyplot(fig)

    st.subheader("4️⃣ Noise Estimation")
    rms = np.sqrt(np.mean(y**2))        # Root Mean Square energy
    db = 20 * np.log10(rms)             # Convert to dB

    st.metric(label="Estimated Noise Level", value=f"{db:.2f} dB")
