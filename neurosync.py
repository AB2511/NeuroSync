import streamlit as st
import numpy as np
import pandas as pd
import mne
from scipy.signal import welch
import joblib
import os
import tempfile

# Frequency bands
freq_bands = {
    'delta': (0.5, 4),
    'theta': (4, 8),
    'alpha': (8, 12),
    'beta': (12, 30),
    'gamma': (30, 40)
}

# Load model from uploaded file
@st.cache_resource
def load_model(uploaded_model=None):
    try:
        if uploaded_model is None:
            raise ValueError("Please upload a valid .pkl file.")
        with tempfile.NamedTemporaryFile(delete=False, suffix='.pkl') as tmp:
            tmp.write(uploaded_model.read())
            tmp_path = tmp.name
        try:
            model = joblib.load(tmp_path)
            return model
        finally:
            os.unlink(tmp_path)  # Clean up
    except Exception as e:
        st.error(f"Failed to load model: {str(e)}. Ensure the .pkl file is a valid Random Forest model.")
        return None

# Preprocessing function
def preprocess_eeg(data, sfreq=128):
    try:
        raw = mne.io.RawArray(data, mne.create_info(21, sfreq, ch_types='eeg'), verbose=False)
        raw.filter(0.5, 40, verbose=False)
        ica = mne.preprocessing.ICA(n_components=15, random_state=42, max_iter=200)
        ica.fit(raw, verbose=False)
        raw = ica.apply(raw, verbose=False)
        
        data, _ = raw[:, :]
        features = []
        band_powers = []
        for ch in range(data.shape[0]):
            f, psd = welch(data[ch], fs=sfreq, nperseg=data.shape[1])
            channel_powers = []
            for band, (low, high) in freq_bands.items():
                power = np.mean(psd[(f >= low) & (f < high)])
                channel_powers.append(power)
                features.append(power)
            band_powers.append(channel_powers)
            features.append(np.mean(data[ch]))
            features.append(np.var(data[ch]))
        for ch_powers in band_powers:
            theta = ch_powers[1]
            beta = ch_powers[3]
            alpha = ch_powers[2]
            features.append(theta / beta if beta != 0 else 0)
            features.append(alpha / beta if beta != 0 else 0)
        return np.nan_to_num(features, nan=0.0)
    except Exception as e:
        st.error(f"Error in preprocessing: {str(e)}")
        return None

# Streamlit app
st.title("NeuroSync: A BCI-Powered Focus Assistant")
st.write("Upload EEG data to detect focus state (simulated BCI input).")
st.write("Developed by Anjali, July 2025.")
st.write("**Usage Tip**: Upload a 21-channel EEG CSV (e.g., sample_eeg.csv) for analysis.")

# Clear cache button
if st.button("Clear Cache"):
    st.cache_resource.clear()
    st.success("Cache cleared. Please reload the model.")

# Model input
st.subheader("Load Model")
uploaded_model = st.file_uploader("Upload focus_classifier_rf.pkl", type="pkl")
model = None
if uploaded_model:
    model = load_model(uploaded_model=uploaded_model)
if model:
    st.success("Model loaded successfully!")

# Upload EEG data
uploaded_file = st.file_uploader("Upload EEG CSV (21 channels, e.g., sample_eeg.csv)", type="csv")
if uploaded_file and model:
    try:
        eeg_data = pd.read_csv(uploaded_file).values.T  # Transpose to match [channels, samples]
        if eeg_data.shape[0] != 21:
            st.error("EEG data must have 21 channels.")
        else:
            features = preprocess_eeg(eeg_data)
            if features is not None:
                prediction = model.predict([features])[0]
                state = "Focused" if prediction == 1 else "Distracted"
                st.write(f"**Focus State**: {state}")
                st.write("Model Accuracy: 94% (based on EEGMAT dataset)")
                
                # Confidence visualization
                st.subheader("Focus Prediction Confidence")
                confidence = model.predict_proba([features])[0]
                confidence_df = pd.DataFrame({
                    "State": ["Distracted", "Focused"],
                    "Confidence": confidence
                })
                st.bar_chart(confidence_df.set_index("State"))
                
                # Time-series visualization
                sfreq = 128
                window_size = 2 * sfreq
                step_size = sfreq // 2  # Smoother trend
                states = []
                times = []
                for i in range(0, eeg_data.shape[1] - window_size + 1, step_size):
                    window = eeg_data[:, i:i + window_size]
                    if window.shape[1] == window_size:
                        features = preprocess_eeg(window, sfreq)
                        if features is not None:
                            states.append(1 if model.predict([features])[0] == 1 else 0)
                            times.append(i / sfreq)
                if states:
                    st.subheader("Focus Trend Over Time")
                    trend_df = pd.DataFrame({"Time (s)": times, "Focus State (1=Focused, 0=Distracted)": states})
                    st.line_chart(trend_df.set_index("Time (s)"))
                
                # Frequency band power visualization
                st.subheader("Frequency Band Powers (Channel 1)")
                f, psd = welch(eeg_data[0], fs=sfreq, nperseg=window_size)
                band_powers = {band: np.mean(psd[(f >= low) & (f < high)]) for band, (low, high) in freq_bands.items()}
                band_df = pd.DataFrame({"Band": list(band_powers.keys()), "Power": list(band_powers.values())})
                st.bar_chart(band_df.set_index("Band"))
    except Exception as e:
        st.error(f"Error processing EEG data: {str(e)}")

# Project description
st.subheader("About NeuroSync")
st.write("""
NeuroSync uses a Random Forest model trained on the EEGMAT dataset (72 EDF files, 36 subjects) to classify brain activity as 'Focused' or 'Distracted' with 94% accuracy. 
Features include power in delta, theta, alpha, beta, and gamma bands, plus statistical measures and band ratios. 
Future versions will integrate real-time BCI hardware (e.g., Muse, NeuroSky) for live focus monitoring.
Developed by Anjali, July 2025.
""")
