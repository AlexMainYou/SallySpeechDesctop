import sys
from PyQt6.QtWidgets import *
from PyQt6.QtCore import *
from PyQt6.QtGui import *
import pyaudio
import numpy as np
import threading
import time
from pynput import keyboard
from pynput.keyboard import Controller, Key
import warnings
import requests
import wave
import os
import subprocess
import uuid

warnings.filterwarnings("ignore", category=FutureWarning)

FIREWORKS_API_KEY = "YOUR_FIREWORKS_API_KEY"

class RecordingIndicator(QWidget):
    def __init__(self):
        super().__init__()
        self.is_recording = False
        self.setFixedSize(20, 20)

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)

        color = Qt.GlobalColor.red if self.is_recording else Qt.GlobalColor.gray
        painter.setBrush(QBrush(color))
        painter.setPen(Qt.PenStyle.NoPen)
        painter.drawEllipse(2, 2, 16, 16)

    def set_recording(self, state):
        self.is_recording = state
        self.update()

class WhisperGUI(QMainWindow):
    update_transcript = pyqtSignal(str)

    def __init__(self):
        super().__init__()
        self.initUI()
        self.setup_audio()

    def initUI(self):
        self.setWindowTitle('Whisper Transcriber')
        self.setFixedSize(400, 200)

        central_widget = QWidget()
        self.setCentralWidget(central_widget)

        layout = QVBoxLayout(central_widget)

        top_panel = QHBoxLayout()

        self.indicator = RecordingIndicator()
        top_panel.addWidget(self.indicator)

        self.status_label = QLabel('Нажмите Insert для начала/остановки записи')
        top_panel.addWidget(self.status_label)
        top_panel.addStretch()

        copy_button = QPushButton('Копировать')
        copy_button.setFixedWidth(100)
        copy_button.clicked.connect(self.copy_text)
        top_panel.addWidget(copy_button)

        layout.addLayout(top_panel)

        self.text_edit = QTextEdit()
        self.text_edit.setReadOnly(True)
        layout.addWidget(self.text_edit)

        self.update_transcript.connect(self.update_transcript_text)

        self.setStyleSheet("""
            QMainWindow {
                background-color: #202020;
            }
            QWidget {
                background-color: #202020;
                color: #ffffff;
            }
            QTextEdit {
                background-color: #202020;
                color: #ffffff;
                border: 1px solid #3d3d3d;
                border-radius: 5px;
                padding: 5px;
            }
            QLabel {
                color: #ffffff;
            }
            QPushButton {
                background-color: #202020;
                color: #ffffff;
                border: 1px solid #3d3d3d;
                border-radius: 3px;
                padding: 5px 10px;
                min-width: 80px;
            }
            QPushButton:hover {
                background-color: #202020;
                border: 1px solid #4d4d4d;
            }
            QPushButton:pressed {
                background-color: #202020;
            }
        """)

    def setup_audio(self):
        self.kbd = Controller()
        self.is_recording = False
        self.audio_buffer = []
        self.program_running = True

        self.CHUNK = 1024
        self.FORMAT = pyaudio.paInt16
        self.CHANNELS = 1
        self.RATE = 48000

        self.p = pyaudio.PyAudio()

        self.keyboard_listener = keyboard.Listener(on_press=self.on_press)
        self.keyboard_listener.start()

        self.record_thread = threading.Thread(target=self.record_audio)
        self.record_thread.daemon = True
        self.record_thread.start()

    def on_press(self, key):
        if key == keyboard.Key.insert:
            self.toggle_recording()
        elif key == keyboard.Key.end:
            self.program_running = False
            self.close()
            return False

    def toggle_recording(self):
        self.is_recording = not self.is_recording
        self.indicator.set_recording(self.is_recording)
        status = "Запись идет..." if self.is_recording else "Запись остановлена"
        self.status_label.setText(status)

        if not self.is_recording and self.audio_buffer:
            self.process_audio_buffer()

    def process_audio_buffer(self):
        # Создаем папку temp в корневой директории программы, если её нет
        temp_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "temp")
        os.makedirs(temp_dir, exist_ok=True)
      
        audio_filename = os.path.join(temp_dir, f"audio_{uuid.uuid4()}.mp3")

        # Сохраняем сырой PCM в WAV для последующего сжатия
        raw_wav_filename = os.path.join(temp_dir, f"raw_{os.path.basename(audio_filename)}.wav")
        with wave.open(raw_wav_filename, 'wb') as wf:
            wf.setnchannels(self.CHANNELS)
            wf.setsampwidth(self.p.get_sample_size(self.FORMAT))
            wf.setframerate(self.RATE)
            wf.writeframes(b''.join(self.audio_buffer))

        self.audio_buffer = []

        # Сжимаем WAV в MP3 с помощью FFmpeg
        try:
            subprocess.run([
                os.path.join(os.path.dirname(os.path.abspath(__file__)), 'ffmpeg', 'ffmpeg.exe'), # Полный путь до ffmpeg
                '-y',
                '-f', 's16le',
                '-ar', str(self.RATE),
                '-ac', str(self.CHANNELS),
                '-i', raw_wav_filename,
                '-c:a', 'libmp3lame',
                '-q:a', '2',
                audio_filename
            ], check=True, capture_output=True, creationflags=subprocess.CREATE_NO_WINDOW)

            with open(audio_filename, 'rb') as f:
              audio_data = f.read()

            threading.Thread(target=self.send_for_transcription, args=(audio_data, audio_filename)).start()

        except subprocess.CalledProcessError as e:
            print(f"Ошибка при сжатии аудио: {e}")
            self.update_transcript.emit(f"Ошибка при сжатии аудио: {e}")
        finally:
            os.remove(raw_wav_filename)

    def send_for_transcription(self, audio_data, audio_filename):
        try:
            with open(audio_filename, 'rb') as f:
                response = requests.post(
                    "https://audio-prod.us-virginia-1.direct.fireworks.ai/v1/audio/transcriptions",
                    headers={"Authorization": f"Bearer {FIREWORKS_API_KEY}"},
                    files={"file": f},
                    data={
                        "model": "whisper-v3",
                        "temperature": "0.2",
                        "vad_model": "silero",
                        "language": "ru"
                    },
                )

            if response.status_code == 200:
                result = response.json()
                transcribed_text = result.get('text', '').strip()
                if transcribed_text:
                    self.update_transcript.emit(transcribed_text)
                    self.simulate_typing(transcribed_text + " ")
            else:
                error_message = f"Error: {response.status_code} - {response.text}"
                print(error_message)
                self.update_transcript.emit(error_message)

        except Exception as e:
            error_message = f"Error during transcription: {str(e)}"
            print(error_message)
            self.update_transcript.emit(error_message)

    def record_audio(self):
        stream = None
        try:
            stream = self.p.open(format=self.FORMAT,
                                channels=self.CHANNELS,
                                rate=self.RATE,
                                input=True,
                                frames_per_buffer=self.CHUNK)

            while self.program_running:
                if self.is_recording:
                    try:
                        data = stream.read(self.CHUNK, exception_on_overflow=False)
                        self.audio_buffer.append(data)
                    except IOError as e:
                        print(f"IOError during recording: {e}")
                        if 'Input overflowed' in str(e):
                            print("Input overflowed. Resetting stream.")
                            stream.stop_stream()
                            stream.close()
                            stream = self.p.open(format=self.FORMAT,
                                channels=self.CHANNELS,
                                rate=self.RATE,
                                input=True,
                                frames_per_buffer=self.CHUNK)
                        continue
                time.sleep(0.01)
        except Exception as e:
            print(f"Error during audio setup: {e}")
        finally:
            if stream:
                stream.stop_stream()
                stream.close()

    def copy_text(self):
        clipboard = QApplication.clipboard()
        clipboard.setText(self.text_edit.toPlainText())

    def simulate_typing(self, text):
        for char in text:
            self.kbd.type(char)
            time.sleep(0.01)  # Небольшая задержка между символами

    @pyqtSlot(str)
    def update_transcript_text(self, text):
        self.text_edit.setText(text)

    def closeEvent(self, event):
        self.program_running = False
        self.keyboard_listener.stop()
        self.p.terminate()
        event.accept()

if __name__ == '__main__':
    app = QApplication(sys.argv)
    ex = WhisperGUI()
    ex.show()
    sys.exit(app.exec())