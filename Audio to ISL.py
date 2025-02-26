import tkinter as tk
from tkinter import filedialog
from tkinter import messagebox
import tensorflow as tf
import librosa
import numpy as np

# Load the pre-trained model
model = tf.keras.models.load_model('transformer_model.h5')

# Function to process the audio and make a prediction
def process_audio(file_path):
    try:
        # Load the audio file using librosa
        audio, sr = librosa.load(file_path, sr=None)  # Use the original sample rate
        # Pre-process the audio as per the model's requirement, this is a basic example:
        audio_feature = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=13)  # MFCC features (adjust as needed)
        audio_feature = np.expand_dims(audio_feature, axis=0)  # Add batch dimension

        # Make a prediction
        prediction = model.predict(audio_feature)
        return prediction
    except Exception as e:
        print(f"Error processing audio: {e}")
        return None

# Function to handle file selection and prediction
def upload_audio():
    # Open a file dialog to select an audio file
    file_path = filedialog.askopenfilename(filetypes=[("Audio Files", "*.wav *.mp3 *.flac")])
    if file_path:
        # Call the process_audio function to make a prediction
        prediction = process_audio(file_path)
        if prediction is not None:
            # Show the prediction result
            messagebox.showinfo("Prediction Result", f"Prediction: {prediction}")
        else:
            messagebox.showerror("Error", "There was an error processing the audio file.")
    else:
        messagebox.showwarning("No File", "Please select a valid audio file.")

# Create the Tkinter window
window = tk.Tk()
window.title("Audio Prediction")

# Add a button to upload an audio file and make a prediction
upload_button = tk.Button(window, text="Upload Audio and Predict", command=upload_audio)
upload_button.pack(pady=20)

# Run the Tkinter main loop
window.mainloop()
