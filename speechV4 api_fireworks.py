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
import io
import tempfile

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
        
        # Создаем центральный виджет
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        # Главный layout
        layout = QVBoxLayout(central_widget)
        
        # Верхняя панель с индикатором и статусом
        top_panel = QHBoxLayout()
        
        self.indicator = RecordingIndicator()
        top_panel.addWidget(self.indicator)
        
        self.status_label = QLabel('Нажмите Insert для начала/остановки записи')
        top_panel.addWidget(self.status_label)
        top_panel.addStretch()
        
        # Кнопка копирования
        copy_button = QPushButton('Копировать')
        copy_button.setFixedWidth(100)
        copy_button.clicked.connect(self.copy_text)
        top_panel.addWidget(copy_button)
        
        layout.addLayout(top_panel)
        
        # Поле для текста
        self.text_edit = QTextEdit()
        self.text_edit.setReadOnly(True)
        layout.addWidget(self.text_edit)
        
        # Подключаем сигнал обновления транскрипции
        self.update_transcript.connect(self.update_transcript_text)
        
        # Стилизация
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
        
        # Параметры аудио
        self.CHUNK = 1024
        self.FORMAT = pyaudio.paFloat32
        self.CHANNELS = 1
        self.RATE = 16000
        
        self.p = pyaudio.PyAudio()
        
        # Запуск потоков
        self.keyboard_listener = keyboard.Listener(on_press=self.on_press)
        self.keyboard_listener.start()
        
        self.record_thread = threading.Thread(target=self.record_audio)
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

    def transcribe_audio(self, audio_data):
        # Создаем временный WAV файл
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as temp_wav:
            with wave.open(temp_wav.name, 'wb') as wf:
                wf.setnchannels(self.CHANNELS)
                wf.setsampwidth(4)  # для float32
                wf.setframerate(self.RATE)
                wf.writeframes(audio_data)

        # Отправляем файл в API Fireworks
        try:
            with open(temp_wav.name, 'rb') as f:
                response = requests.post(
                    "https://audio-prod.us-virginia-1.direct.fireworks.ai/v1/audio/transcriptions",
                    headers={"Authorization": f"Bearer {FIREWORKS_API_KEY}"},
                    files={"file": f},
                    data={
                        "model": "whisper-v3",
                        "temperature": "0",
                        "vad_model": "silero"
                    },
                )

            if response.status_code == 200:
                result = response.json()
                return result.get('text', '').strip()
            else:
                print(f"Error: {response.status_code}", response.text)
                return ''

        except Exception as e:
            print(f"Error during transcription: {str(e)}")
            return ''

    def record_audio(self):
        stream = self.p.open(format=self.FORMAT,
                           channels=self.CHANNELS,
                           rate=self.RATE,
                           input=True,
                           frames_per_buffer=self.CHUNK)

        while self.program_running:
            if self.is_recording:
                data = stream.read(self.CHUNK)
                self.audio_buffer.append(data)
            else:
                if self.audio_buffer:
                    audio_data = b''.join(self.audio_buffer)
                    self.audio_buffer = []
                    
                    transcribed_text = self.transcribe_audio(audio_data)
                    
                    if transcribed_text:
                        self.update_transcript.emit(transcribed_text)
                        self.type_text(transcribed_text)
            time.sleep(0.01)
        
        stream.stop_stream()
        stream.close()

    def type_text(self, text):
        self.kbd.press(Key.alt_l)
        self.kbd.press(Key.shift_l)
        self.kbd.release(Key.shift_l)
        self.kbd.release(Key.alt_l)
        time.sleep(0.1)
        
        for char in text:
            self.kbd.type(char)
            time.sleep(0.001)
        
        self.kbd.press(Key.space)
        self.kbd.release(Key.space)
        
        self.kbd.press(Key.alt_l)
        self.kbd.press(Key.shift_l)
        self.kbd.release(Key.shift_l)
        self.kbd.release(Key.alt_l)

    def copy_text(self):
        clipboard = QApplication.clipboard()
        clipboard.setText(self.text_edit.toPlainText())

    @pyqtSlot(str)
    def update_transcript_text(self, text):
        self.text_edit.setText(text)

    def closeEvent(self, event):
        self.program_running = False
        self.p.terminate()
        event.accept()

if __name__ == '__main__':
    app = QApplication(sys.argv)
    ex = WhisperGUI()
    ex.show()
    sys.exit(app.exec())
