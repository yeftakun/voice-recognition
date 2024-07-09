# Voice Recognition

## Dataset

### Rekam dataset
(mic saya burik, jadi rada" pas nyoba):
1. file `enrol.py`, masukan dulu nama individunya.
2. Setelah tekan "Mulai Perekaman" suara langsung direkam. Tekan "Stop" untuk berhenti.
3. Ulangi langkah 2 untuk menambahkan file suara.
4. Upload gambar profile.
5. Simpan data untuk menyimpan individu. Batal untuk menghapus.

### Buat dataset:
1. Audio file yang mengandung voice karakter/individunya
2. Remove noise/bgm di [Vocal Remover](https://vocalremover.org/)
3. Tempatkan dataset seperti dibawah


```
dataset/
│
├── bocchi/
│   ├── voice1.mp3
│   ├── voice2.mp3
│   ├── voice3.mp3
│   └── profile.jpg
│
├── nijika/
│   ├── voice1.mp3
│   ├── voice2.mp3
│   ├── voice3.mp3
│   └── profile.jpg
│
└── susman-from-mic/
    ├── voice1.mp3
    ├── voice2.mp3
    ├── voice3.mp3
    ├── voice4.mp3
    └── profile.jpg

```

### Memulai

1. Sesuaikan file `.env`.
2. Jalankan `training.py`.
3. Pastikan sudah menambahkan file voice input di `audio/` dan sesuaikan `INPUT_FILE` pada `.env`
4. Jalankan `main.py` untuk testing.