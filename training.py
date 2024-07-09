import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
import librosa
import pickle

# Fungsi untuk ekstraksi fitur MFCC dari file audio
def extract_features(file_path, max_pad_len=40):
    try:
        audio, sample_rate = librosa.load(file_path, res_type='kaiser_fast')
        mfccs = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=40)
        
        if mfccs.shape[1] < max_pad_len:
            pad_width = max_pad_len - mfccs.shape[1]
            mfccs = np.pad(mfccs, pad_width=((0, 0), (0, pad_width)), mode='constant')
        else:
            mfccs = mfccs[:, :max_pad_len]
        
        return mfccs
    except Exception as e:
        print(f"Error encountered while parsing file: {file_path}\n{e}")
        return None

# Direktori dataset
dataset_dir = 'dataset'
categories = os.listdir(dataset_dir)

# Menyiapkan data dan label
X, y = [], []

for category in categories:
    category_path = os.path.join(dataset_dir, category)
    if os.path.isdir(category_path):
        for file_name in os.listdir(category_path):
            if file_name.endswith('.mp3'):
                file_path = os.path.join(category_path, file_name)
                features = extract_features(file_path)
                if features is not None:
                    X.append(features)
                    y.append(category)

# Konversi data dan label menjadi numpy array
X = np.array(X)
y = np.array(y)

if len(X) == 0:
    raise ValueError("No valid audio files found. Please check the dataset directory.")

# Encode labels
label_encoder = {label: index for index, label in enumerate(np.unique(y))}
y_encoded = np.array([label_encoder[label] for label in y])

# Simpan label encoder
with open('label_encoder.pkl', 'wb') as f:
    pickle.dump(label_encoder, f)

# Model LSTM
model = Sequential([
    tf.keras.Input(shape=(40, 40)),  # Menggunakan Input layer dengan shape yang benar
    LSTM(128, return_sequences=True),
    LSTM(128),
    Dense(128, activation='relu'),
    Dense(len(label_encoder), activation='softmax')
])

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Training model
model.fit(X, y_encoded, epochs=500, batch_size=16, validation_split=0.2)

# Simpan model
model.save('voice_recognition_model.h5')
