# Import library yang dibutuhkan
import numpy as np
import librosa
import os
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout
from sklearn.model_selection import train_test_split

# Fungsi untuk memuat data suara dan mengekstrak fitur MFCC
def load_audio_files(file_paths, max_len=50):
    data = []
    for file_path in file_paths:
        signal, sr = librosa.load(file_path, sr=22050)
        mfccs = librosa.feature.mfcc(signal, sr=sr, n_mfcc=13)
        pad_width = max_len - mfccs.shape[1]
        mfccs = np.pad(mfccs, pad_width=((0, 0), (0, pad_width)), mode='constant')
        data.append(mfccs)
    return np.array(data)

# Daftar file suara (file wav)
file_paths = ['audio/model/audio1.wav', 'audio/model/audio2.wav']

# Label (harus diatur sesuai dengan file audio)
labels = [0, 1, 0]  # Misal 0 untuk kelas pertama, 1 untuk kelas kedua

# Memuat data dan label
X = load_audio_files(file_paths)
y = np.array(labels)

# Split data menjadi training dan testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Membuat model
model = Sequential()
model.add(LSTM(128, input_shape=(X_train.shape[1], X_train.shape[2]), return_sequences=True))
model.add(Dropout(0.2))
model.add(LSTM(128))
model.add(Dropout(0.2))
model.add(Dense(1, activation='sigmoid'))

# Compile model
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# Melatih model
model.fit(X_train, y_train, epochs=50, batch_size=32, validation_data=(X_test, y_test))

# Menyimpan model
model.save('voice_recognition_model.h5')
