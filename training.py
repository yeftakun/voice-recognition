import numpy as np
import librosa
import os
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout
from sklearn.model_selection import train_test_split

# Function to load audio files and extract MFCC features
def load_audio_files(file_paths, max_len=50):
    data = []
    for file_path in file_paths:
        # Load audio file
        signal, sr = librosa.load(file_path, sr=22050, mono=True)  # Ensure mono=True for single-channel audio
        
        # Extract MFCC features
        mfccs = librosa.feature.mfcc(y=signal, sr=sr, n_mfcc=13)
        
        # Pad or truncate mfccs to max_len
        if mfccs.shape[1] > max_len:
            mfccs = mfccs[:, :max_len]
        else:
            pad_width = max_len - mfccs.shape[1]
            mfccs = np.pad(mfccs, pad_width=((0, 0), (0, pad_width)), mode='constant')
        
        data.append(mfccs.T)  # Transpose to match LSTM input shape (time_steps, features)
    
    return np.array(data)

# List of audio files (wav files)
file_paths = ['voice1.mp3']  # Update with your actual file paths

# Labels (must match the number of audio files)
labels = [0, 1, 0]  # For example, 0 for class 1, 1 for class 2

# Load data and labels
X = load_audio_files(file_paths)
y = np.array(labels)

# Split data into training and testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Build the model
model = Sequential()
model.add(LSTM(128, input_shape=(X_train.shape[1], X_train.shape[2]), return_sequences=True))
model.add(Dropout(0.2))
model.add(LSTM(128))
model.add(Dropout(0.2))
model.add(Dense(1, activation='sigmoid'))

# Compile the model
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# Train the model
model.fit(X_train, y_train, epochs=50, batch_size=32, validation_data=(X_test, y_test))

# Save the model
model.save('voice_recognition_model.h5')
