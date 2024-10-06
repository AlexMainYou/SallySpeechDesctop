import whisper
import pyaudio
import numpy as np
import threading
import time
from pynput import keyboard
from pynput.keyboard import Controller, Key

# Load the Whisper Turbo model
model = whisper.load_model("large-v3-turbo", device="cuda")

# Initialize the keyboard controller
kbd = Controller()

# Audio recording parameters
CHUNK = 1024
FORMAT = pyaudio.paFloat32
CHANNELS = 1
RATE = 16000

# Initialize PyAudio
p = pyaudio.PyAudio()

# Global variables
is_recording = False
audio_buffer = []
program_running = True

def toggle_recording():
    global is_recording
    is_recording = not is_recording
    print("Запись:", "Начата" if is_recording else "Остановлена")

def on_press(key):
    global program_running
    if key == keyboard.Key.insert:
        toggle_recording()
    elif key == keyboard.Key.end:
        program_running = False
        print("Выход из программы...")
        return False

def type_text(text):
    # Переключаемся на русскую раскладку
    kbd.press(Key.alt_l)
    kbd.press(Key.shift_l)
    kbd.release(Key.shift_l)
    kbd.release(Key.alt_l)
    time.sleep(0.1)  # Небольшая задержка для переключения раскладки
    
    for char in text:
        kbd.type(char)
        time.sleep(0.001)  # 1ms delay between keypresses
    
    kbd.press(Key.space)
    kbd.release(Key.space)
    
    # Возвращаемся на английскую раскладку
    kbd.press(Key.alt_l)
    kbd.press(Key.shift_l)
    kbd.release(Key.shift_l)
    kbd.release(Key.alt_l)

def record_audio():
    stream = p.open(format=FORMAT,
                    channels=CHANNELS,
                    rate=RATE,
                    input=True,
                    frames_per_buffer=CHUNK)

    global is_recording, audio_buffer
    while program_running:
        if is_recording:
            data = stream.read(CHUNK)
            audio_buffer.append(data)
        else:
            if audio_buffer:
                audio_data = b''.join(audio_buffer)
                audio_array = np.frombuffer(audio_data, dtype=np.float32)
                audio_buffer = []  # Clear the buffer
                
                # Normalize audio to the range [-1, 1]
                audio_array = audio_array / np.max(np.abs(audio_array))
                
                # Transcribe using Whisper Turbo
                result = model.transcribe(audio_array, language="ru")
                transcribed_text = result["text"].strip()
                
                if transcribed_text:
                    print(f"Транскрибировано: {transcribed_text}")
                    type_text(transcribed_text)
        time.sleep(0.01)
    
    stream.stop_stream()
    stream.close()

# Start the keyboard listener
listener = keyboard.Listener(on_press=on_press)
listener.start()

# Start the recording in a separate thread
record_thread = threading.Thread(target=record_audio)
record_thread.start()

print("Нажмите Insert для начала/остановки записи. Нажмите End для выхода.")

while program_running:
    time.sleep(0.1)

p.terminate()