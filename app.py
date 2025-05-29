from flask import Flask, request, jsonify
from flask_cors import CORS  # Tambahkan ini
import numpy as np
import librosa
import pywt
import os
from tensorflow.keras.models import load_model
from scipy.stats import entropy, skew, kurtosis

app = Flask(__name__)

# Aktifkan CORS untuk semua rute
CORS(app)

# Load the trained deep learning model
model = load_model('dnn_model_fold_2.h5')

def extract_features(file_path):
    try:
        y, sr = librosa.load(file_path, duration=2.5, sr=22050)
        if y.size == 0:
            raise ValueError("Audio data is empty or too short.")
        
        # MFCCs
        mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)
        mfccs_means = np.mean(mfccs, axis=1)
        mfccs_std = np.std(mfccs, axis=1)
        mfccs_max = np.max(mfccs, axis=1)
        mfccs_min = np.min(mfccs, axis=1)

        entropy_values = [entropy(np.histogram(mfccs[i])[0]) for i in range(mfccs.shape[0])]

        coeffs = pywt.wavedec(y, 'db1', level=4)
        wavelet_features = []
        for coeff in coeffs:
            if coeff.size > 0:
                wavelet_features.append(np.array([np.mean(coeff), np.std(coeff), np.max(coeff), np.min(coeff)]))
        if len(wavelet_features) == 0:
            raise ValueError("Wavelet features cannot be calculated.")
        
        wavelet_features = np.concatenate(wavelet_features, axis=0)

        wavelet_means = np.mean(wavelet_features)
        wavelet_std = np.std(wavelet_features)
        wavelet_max = np.max(wavelet_features)
        wavelet_min = np.min(wavelet_features)

        med_mfcc = np.median(mfccs, axis=1)
        var_mfcc = np.var(mfccs, axis=1)
        skew_mfcc = skew(mfccs, axis=1)
        q1_mfcc = np.percentile(mfccs, 25, axis=1)
        q3_mfcc = np.percentile(mfccs, 75, axis=1)
        iqr_mfcc = q3_mfcc - q1_mfcc
        minmax_mfcc = np.ptp(mfccs, axis=1)
        kurt_mfcc = kurtosis(mfccs, axis=1)

        med_wavelet = np.median(wavelet_features)
        var_wavelet = np.var(wavelet_features)
        skew_wavelet = skew(wavelet_features)
        q1_wavelet = np.percentile(wavelet_features, 25)
        q3_wavelet = np.percentile(wavelet_features, 75)
        iqr_wavelet = q3_wavelet - q1_wavelet
        minmax_wavelet = np.ptp(wavelet_features)
        kurt_wavelet = kurtosis(wavelet_features)

        features = np.concatenate([
            mfccs_means, mfccs_std, mfccs_max, mfccs_min,
            entropy_values, [wavelet_means, wavelet_std, wavelet_max, wavelet_min],
            med_mfcc, var_mfcc, skew_mfcc, q1_mfcc, q3_mfcc, iqr_mfcc, minmax_mfcc, kurt_mfcc,
            [med_wavelet, var_wavelet, skew_wavelet, q1_wavelet, q3_wavelet, iqr_wavelet, minmax_wavelet, kurt_wavelet]
        ])

        return features

    except Exception as e:
        print(f"Error extracting features from {file_path}: {e}")
        return np.array([])

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Cek apakah ada file audio dalam request
        if 'audio' not in request.files:
            return jsonify({'error': 'No audio file provided'}), 400

        # Ambil file audio dari request
        file = request.files['audio']

        # Validasi apakah file memiliki format .wav
        if not file.filename.endswith('.wav'):
            return jsonify({'error': 'Invalid file format. Please upload a .wav file.'}), 400

        # Tentukan path penyimpanan file di folder 'temp'
        if not os.path.exists('temp'):
            os.makedirs('temp')  # Buat folder 'temp' jika belum ada
        file_path = os.path.join('temp', file.filename)

        # Simpan file di folder 'temp'
        file.save(file_path)

        # Ekstraksi fitur dari file audio
        features = extract_features(file_path)
        print("Extracted features:", features)
        print("Feature size:", features.size)
        if features.size == 0:
            return jsonify({'error': 'Error extracting features from audio file'}), 500

        # Reshape fitur untuk model CNN
        features = features.reshape(1, features.shape[0], 1)

        # Prediksi menggunakan model
        prediction = model.predict(features)[0][0]

        # Hapus file setelah selesai digunakan
        os.remove(file_path)

        # Return hasil prediksi
        result = 'MI Detected' if prediction >= 0.5 else 'Normal'
        print("Raw prediction:", prediction)
        return jsonify({'result': result})

    except Exception as e:
        # Tangani jika terjadi error pada proses
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
