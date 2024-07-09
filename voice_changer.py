# Import library yang dibutuhkan
import numpy as np
import librosa
import soundfile as sf
import tensorflow as tf

# Memuat model yang sudah dilatih
model = tf.keras.models.load_model('audio/model/voice_recognition_model.h5')

# Fungsi untuk memuat data suara dan mengekstrak fitur MFCC
def extract_mfcc(file_path, max_len=50):
    signal, sr = librosa.load(file_path, sr=22050)
    mfccs = librosa.feature.mfcc(signal, sr=sr, n_mfcc=13)
    pad_width = max_len - mfccs.shape[1]
    mfccs = np.pad(mfccs, pad_width=((0, 0), (0, pad_width)), mode='constant')
    return np.array(mfccs)

# Fungsi untuk mengubah suara berdasarkan model
def transform_voice(input_file, output_file):
    mfccs = extract_mfcc(input_file)
    mfccs = mfccs.reshape(1, mfccs.shape[0], mfccs.shape[1])
    
    # Melakukan prediksi dengan model
    predicted_label = model.predict(mfccs)
    
    # Proses transformasi suara dapat disesuaikan berdasarkan prediksi
    # Di sini hanya memberikan contoh sederhana
    signal, sr = librosa.load(input_file, sr=22050)
    transformed_signal = signal * predicted_label[0][0]
    
    # Menyimpan suara yang sudah diubah
    sf.write(output_file, transformed_signal, sr)

# Contoh penggunaan
transform_voice('audio/input/input_audio.wav', 'audio/output/output_audio.wav')
