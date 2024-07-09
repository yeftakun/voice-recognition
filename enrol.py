# On-going
import os
import wave
import pyaudio
import tkinter as tk
from tkinter import filedialog, messagebox

class VoiceRecorderApp:
    def __init__(self, master):
        self.master = master
        self.master.title("Voice Enrolment")
        self.master.geometry("400x400")

        self.recording = False
        self.frames = []
        self.p = pyaudio.PyAudio()

        self.create_widgets()

    def create_widgets(self):
        self.label = tk.Label(self.master, text="Nama Individu:")
        self.label.pack(pady=10)

        self.name_entry = tk.Entry(self.master)
        self.name_entry.pack(pady=5)

        self.record_button = tk.Button(self.master, text="Mulai Perekaman", command=self.start_recording)
        self.record_button.pack(pady=5)

        self.stop_button = tk.Button(self.master, text="Stop", command=self.stop_recording)
        self.stop_button.pack(pady=5)

        self.upload_button = tk.Button(self.master, text="Upload Profile Image", command=self.upload_image)
        self.upload_button.pack(pady=5)

        self.save_button = tk.Button(self.master, text="Simpan Data", command=self.save_data)
        self.save_button.pack(pady=5)

        self.cancel_button = tk.Button(self.master, text="Batal", command=self.cancel_data)
        self.cancel_button.pack(pady=5)

    def start_recording(self):
        if not self.recording:
            self.recording = True
            self.frames = []
            self.stream = self.p.open(format=pyaudio.paInt16,
                                      channels=1,
                                      rate=44100,
                                      input=True,
                                      frames_per_buffer=1024)
            self.record_audio()

    def record_audio(self):
        if self.recording:
            data = self.stream.read(1024)
            self.frames.append(data)
            self.master.after(1, self.record_audio)

    def stop_recording(self):
        if self.recording:
            self.recording = False
            self.stream.stop_stream()
            self.stream.close()

            name = self.name_entry.get().strip()
            if not name:
                messagebox.showerror("Error", "Masukkan nama individu!")
                return

            category_path = os.path.join('dataset', name)
            if not os.path.exists(category_path):
                os.makedirs(category_path)

            file_index = len([f for f in os.listdir(category_path) if f.endswith('.mp3')]) + 1
            file_path = os.path.join(category_path, f'voice{file_index}.mp3')

            wf = wave.open(file_path, 'wb')
            wf.setnchannels(1)
            wf.setsampwidth(self.p.get_sample_size(pyaudio.paInt16))
            wf.setframerate(44100)
            wf.writeframes(b''.join(self.frames))
            wf.close()

            messagebox.showinfo("Info", f"voice{file_index}.mp3 berhasil disimpan.")

    def upload_image(self):
        name = self.name_entry.get().strip()
        if not name:
            messagebox.showerror("Error", "Masukkan nama individu!")
            return

        category_path = os.path.join('dataset', name)
        if not os.path.exists(category_path):
            os.makedirs(category_path)

        file_path = filedialog.askopenfilename(filetypes=[("Image Files", "*.jpg;*.jpeg;*.png")])
        if file_path:
            profile_path = os.path.join(category_path, 'profile.jpg')
            os.rename(file_path, profile_path)
            messagebox.showinfo("Info", "Profile image berhasil diupload.")

    def save_data(self):
        messagebox.showinfo("Info", "Data berhasil disimpan.")

    def cancel_data(self):
        name = self.name_entry.get().strip()
        if not name:
            messagebox.showerror("Error", "Masukkan nama individu!")
            return

        category_path = os.path.join('dataset', name)
        if os.path.exists(category_path):
            for file_name in os.listdir(category_path):
                file_path = os.path.join(category_path, file_name)
                os.remove(file_path)
            os.rmdir(category_path)

        self.name_entry.delete(0, tk.END)
        messagebox.showinfo("Info", "Data individu berhasil dihapus.")

    def on_closing(self):
        if self.recording:
            self.recording = False
            self.stream.stop_stream()
            self.stream.close()
        self.p.terminate()
        self.master.destroy()

if __name__ == "__main__":
    root = tk.Tk()
    app = VoiceRecorderApp(root)
    root.protocol("WM_DELETE_WINDOW", app.on_closing)
    root.mainloop()
