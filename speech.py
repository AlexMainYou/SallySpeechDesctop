import whisper
import pyaudio
import numpy as np
import threading
import time
from pynput import keyboard
from pynput.keyboard import Controller

# Load the Whisper model
model = whisper.load_model("base")

# Initialize the keyboard controller
kbd = Controller()

# Audio recording parameters
CHUNK = 1024
FORMAT = pyaudio.paFloat32
CHANNELS = 1
RATE = 16000
BUFFER_SECONDS = 7  # Increased from 2 to 7 seconds

# Initialize PyAudio
p = pyaudio.PyAudio()

# Global variables
is_recording = False
audio_buffer = []

def toggle_recording():
    global is_recording
    is_recording = not is_recording
    print("Запись:", "Начата" if is_recording else "Остановлена")

def on_press(key):
    if key == keyboard.Key.insert:
        toggle_recording()

def record_audio():
    stream = p.open(format=FORMAT,
                    channels=CHANNELS,
                    rate=RATE,
                    input=True,
                    frames_per_buffer=CHUNK)

    global is_recording, audio_buffer
    while True:
        if is_recording:
            data = stream.read(CHUNK)
            audio_buffer.append(data)
            
            if len(audio_buffer) > int(RATE / CHUNK * BUFFER_SECONDS):
                audio_data = b''.join(audio_buffer)
                audio_array = np.frombuffer(audio_data, dtype=np.float32)
                audio_buffer = []  # Clear the buffer
                
                # Normalize audio to the range [-1, 1]
                audio_array = audio_array / np.max(np.abs(audio_array))
                
                result = model.transcribe(audio_array, language="ru")
                transcribed_text = result["text"].strip()
                
                if transcribed_text:
                    print(f"Транскрибировано: {transcribed_text}")
                    for char in transcribed_text:
                        kbd.press(char)
                        kbd.release(char)
                        time.sleep(0.001)  # 1ms delay between keypresses
                    kbd.press(' ')
                    kbd.release(' ')
        else:
            time.sleep(0.1)

# Start the keyboard listener
listener = keyboard.Listener(on_press=on_press)
listener.start()

# Start the recording in a separate thread
record_thread = threading.Thread(target=record_audio)
record_thread.start()

print("Нажмите Insert для начала/остановки записи. Нажмите Ctrl+C для выхода.")

try:
    while True:
        time.sleep(0.1)
except KeyboardInterrupt:
    print("Выход из программы...")

p.terminate()