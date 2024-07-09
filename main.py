import os
import numpy as np
import tensorflow as tf
import librosa
import pickle
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from dotenv import load_dotenv

load_dotenv()
model_path = os.getenv('MODEL_PATH')
label_encoder_path = os.getenv('LABEL_ENCODER_PATH')
max_pad_len = int(os.getenv('MAX_PAD_LEN'))
input_file = os.getenv('INPUT_FILE')

# Fungsi untuk ekstraksi fitur MFCC dari file audio
def extract_features(file_path, max_pad_len):
    try:
        audio, sample_rate = librosa.load(file_path, res_type='kaiser_fast')
        mfccs = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=40)
        
        # Melakukan padding jika jumlah fitur kurang dari max_pad_len
        if mfccs.shape[1] < max_pad_len:
            pad_width = max_pad_len - mfccs.shape[1]
            mfccs = np.pad(mfccs, pad_width=((0, 0), (0, pad_width)), mode='constant')
        else:
            mfccs = mfccs[:, :max_pad_len]
        
        return mfccs
    except Exception as e:
        print(f"Error encountered while parsing file: {file_path}")
        print(e)
        return None

# Load model dan label encoder
model = tf.keras.models.load_model(model_path)

with open(label_encoder_path, 'rb') as f:
    label_encoder = pickle.load(f)
    
reverse_label_encoder = {index: label for label, index in label_encoder.items()}

# Path untuk input file
input_file_path = "audio/" + input_file
features = extract_features(input_file_path, max_pad_len=40)  # Sesuaikan max_pad_len sesuai dengan yang digunakan saat training

if features is not None:
    features = np.expand_dims(features, axis=0)  # Menambahkan dimensi batch_size
    features = np.expand_dims(features, axis=-1)  # Menambahkan dimensi channel (untuk grayscale)
    
    prediction = model.predict(features)
    predicted_index = np.argmax(prediction, axis=1)[0]
    predicted_label = reverse_label_encoder[predicted_index]
    
    # Tampilkan hasil prediksi
    print(f"Predicted Label: {predicted_label}")
    
    # Path untuk profile image
    profile_image_path = os.path.join('dataset', predicted_label, 'profile.jpg')
    
    # Tampilkan gambar profil jika ada
    if os.path.exists(profile_image_path):
        img = mpimg.imread(profile_image_path)
        imgplot = plt.imshow(img)
        plt.title(predicted_label)
        plt.axis('off')  # Hapus sumbu plot
        plt.show()
    else:
        print(f"Profile image not found for {predicted_label}")
else:
    print("Could not extract features from input file.")
