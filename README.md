# NeuroSync: A BCI-Powered Focus Assistant

## Overview
NeuroSync is an innovative machine learning application that classifies EEG (electroencephalogram) signals as "Focused" or "Distracted" using a Random Forest model trained on the EEGMAT dataset (72 EDF files from 36 subjects). Achieving 94% accuracy, this project simulates brain-computer interface (BCI) input with pre-recorded EEG data, with future plans to integrate real-time BCI hardware such as Muse or NeuroSky. Developed by Anjali in July 2025, NeuroSync highlights expertise in EEG signal processing, machine learning, and web application development.

- **[Live Demo](https://neurosyncafocusassistant.streamlit.app/)**
- **[GitHub Repository](https://github.com/AB2511/NeuroSync)**

## Features
- **Focus Detection**: Accurately classifies EEG data into "Focused" or "Distracted" states with 94% accuracy.
- **EEG Preprocessing**: Implements bandpass filtering (0.5–40 Hz), Independent Component Analysis (ICA) with 15 components, Welch Power Spectral Density (PSD), and 189 features including delta, theta, alpha, beta, and gamma band powers, mean, variance, and band ratios (theta/beta, alpha/beta).
- **Interactive Web Interface**: Built with Streamlit, featuring visualizations for confidence scores, time-series trends, and frequency band powers.
- **Performance Metrics**: Precision/Recall/F1 scores: 0.96/0.97/0.96 (Distracted), 0.90/0.87/0.89 (Focused).
- **Dataset**: Utilizes the EEGMAT dataset (21 channels, 128 Hz sampling rate).

## Tech Stack
- **Programming Language**: Python
- **EEG Processing**: MNE-Python (filtering, ICA)
- **Machine Learning**: Scikit-learn (Random Forest, SMOTE, hyperparameter tuning)
- **Web Framework**: Streamlit
- **Libraries**: NumPy, Pandas, SciPy, Joblib

## Setup and Installation
To run NeuroSync locally, follow these steps:

1. **Clone the Repository**:
   ```bash
   git clone https://github.com/AB2511/NeuroSync.git
   cd NeuroSync
   ```

2. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the Application**:
   ```bash
   streamlit run neurosync.py
   ```

4. **Upload Required Files**:
   - Download `focus_classifier_rf (1).pkl` and `sample_eeg (1).csv` from [Google Drive link] and upload them via the app interface.

## Usage
- Upload the pre-trained model file `focus_classifier_rf (1).pkl` and a 21-channel EEG CSV file (e.g., `sample_eeg (1).csv`).
- The app will display the focus state ("Focused" or "Distracted"), model accuracy (94%), and interactive visualizations including confidence scores, time-series trends, and frequency band powers.

## Future Work
- Integrate real-time BCI hardware (e.g., Muse, NeuroSky) for live focus monitoring.
- Develop a mobile application to enhance accessibility.
- Explore deep learning models (e.g., CNNs, LSTMs) to improve classification accuracy.

## Project Structure
- `neurosync.py`: Main Streamlit application script.
- `requirements.txt`: List of Python dependencies.
- `.gitignore`: Excludes large files (e.g., `.pkl`, `.csv`) from version control.

## Author
- **Name**: Anjali
- **GitHub**: [https://github.com/AB2511](https://github.com/AB2511)

## License
This project is licensed under the MIT License, allowing for open-source use and modification while retaining the original author's credit.

## Acknowledgments
- Special thanks to the EEGMAT dataset contributors for providing valuable EEG data.
- Inspiration from the xAI and Streamlit communities for tools and support.
