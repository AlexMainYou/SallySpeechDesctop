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

FIREWORKS_API_KEY = "YOUR_FIREWORKS_API_KEY"  # Замените на ваш реальный API ключ

class RecordingIndicator(QWidget):
    def __init__(self):
        super().__init__()
        self.is_recording = False
        self.setFixedSize(24, 24)
        self.animation = QPropertyAnimation(self, b"opacity")
        self.opacity = 1.0
        
    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        color = QColor('#FF4444') if self.is_recording else QColor('#555555')
        painter.setBrush(QBrush(color))
        painter.setPen(QPen(QColor('#333333'), 1))
        painter.setOpacity(self.opacity)
        painter.drawEllipse(2, 2, 20, 20)

    def set_recording(self, state):
        self.is_recording = state
        if state:
            self.animation.setDuration(1000)
            self.animation.setStartValue(0.4)
            self.animation.setEndValue(1.0)
            self.animation.setLoopCount(-1)
            self.animation.start()
        else:
            self.animation.stop()
            self.opacity = 1.0
        self.update()

class WhisperGUI(QMainWindow):
    update_transcript = pyqtSignal(str)

    def __init__(self):
        super().__init__()
        self.initUI()
        self.setup_audio()

    def initUI(self):
        self.setWindowTitle('Whisper Transcriber')
        self.setFixedSize(450, 250)
        self.setWindowIcon(QIcon('microphone.png'))

        central_widget = QWidget()
        self.setCentralWidget(central_widget)

        main_layout = QVBoxLayout(central_widget)
        main_layout.setContentsMargins(10, 10, 10, 10)
        main_layout.setSpacing(10)

        control_panel = QWidget()
        control_layout = QHBoxLayout(control_panel)
        control_layout.setContentsMargins(0, 0, 0, 0)

        self.indicator = RecordingIndicator()
        control_layout.addWidget(self.indicator)

        self.status_label = QLabel('Нажмите Insert для начала записи')
        self.status_label.setFixedHeight(24)
        control_layout.addWidget(self.status_label)
        control_layout.addStretch()

        self.copy_button = QPushButton('Копировать')
        self.copy_button.setFixedSize(90, 30)
        self.copy_button.clicked.connect(self.copy_text)
        control_layout.addWidget(self.copy_button)

        main_layout.addWidget(control_panel)

        self.text_edit = QTextEdit()
        self.text_edit.setReadOnly(True)
        self.text_edit.setFont(QFont('Segoe UI', 10))
        main_layout.addWidget(self.text_edit)

        info_panel = QWidget()
        info_layout = QHBoxLayout(info_panel)
        info_layout.setContentsMargins(5, 2, 5, 2)
        
        self.info_label = QLabel('Готов к записи')
        self.info_label.setStyleSheet('color: #888888; font-size: 11px;')
        info_layout.addWidget(self.info_label)
        info_layout.addStretch()
        
        shortcut_label = QLabel('End - выход')
        shortcut_label.setStyleSheet('color: #888888; font-size: 11px;')
        info_layout.addWidget(shortcut_label)
        
        main_layout.addWidget(info_panel)

        self.update_transcript.connect(self.update_transcript_text)

        self.setStyleSheet("""
    QMainWindow {
        background-color: #1E1E1E;
    }
    QWidget {
        background-color: #1E1E1E;
        color: #FFFFFF;
    }
    QTextEdit {
        background-color: #2A2A2A;
        color: #E0E0E0;
        border: none;
        border-radius: 8px;
        padding: 8px;
        selection-background-color: #404040;
    }
    QTextEdit QScrollBar:vertical {
        border: none;
        background: #2A2A2A;       /* Фон полосы совпадает с QTextEdit */
        width: 10px;
        margin: 0px 0px 0px 0px;
    }
    QTextEdit QScrollBar::handle:vertical {
        background: #505050;        /* Светло-серый ползунок для контраста */
        min-height: 20px;
        border-radius: 5px;
    }
    QTextEdit QScrollBar::add-line:vertical {
        height: 0px;               /* Убираем кнопку "вниз" */
    }
    QTextEdit QScrollBar::sub-line:vertical {
        height: 0px;               /* Убираем кнопку "вверх" */
    }
    QTextEdit QScrollBar::add-page:vertical, QScrollBar::sub-page:vertical {
        background: none;          /* Прозрачный фон за пределами ползунка */
    }
    QLabel {
        color: #FFFFFF;
        font-family: 'Segoe UI';
    }
    QPushButton {
        background-color: #3A3A3A;
        color: #FFFFFF;
        border: none;
        border-radius: 4px;
        padding: 5px 10px;
        font-family: 'Segoe UI';
        font-size: 12px;
    }
    QPushButton:hover {
        background-color: #454545;
    }
    QPushButton:pressed {
        background-color: #303030;
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
        self.RATE = 44100  # Изменил на более стандартное значение

        self.p = pyaudio.PyAudio()
        self.stream = None

        self.keyboard_listener = keyboard.Listener(on_press=self.on_press)
        self.keyboard_listener.start()

        self.record_thread = threading.Thread(target=self.record_audio, daemon=True)
        self.record_thread.start()

    def on_press(self, key):
        try:
            if key == keyboard.Key.insert:
                self.toggle_recording()
            elif key == keyboard.Key.end:
                self.program_running = False
                self.close()
                return False
        except AttributeError:
            pass

    def toggle_recording(self):
        self.is_recording = not self.is_recording
        self.indicator.set_recording(self.is_recording)
        status = "Идет запись..." if self.is_recording else "Нажмите Insert для начала записи"
        info = "Записываю" if self.is_recording else "Готов к записи"
        self.status_label.setText(status)
        self.info_label.setText(info)

        if not self.is_recording and self.audio_buffer:
            threading.Thread(target=self.process_audio_buffer, daemon=True).start()

    def record_audio(self):
        while self.program_running:
            if self.is_recording:
                if not self.stream:
                    self.stream = self.p.open(
                        format=self.FORMAT,
                        channels=self.CHANNELS,
                        rate=self.RATE,
                        input=True,
                        frames_per_buffer=self.CHUNK
                    )
                try:
                    data = self.stream.read(self.CHUNK, exception_on_overflow=False)
                    self.audio_buffer.append(data)
                except IOError as e:
                    self.info_label.setText(f"Ошибка записи: {e}")
            else:
                if self.stream:
                    self.stream.stop_stream()
                    self.stream.close()
                    self.stream = None
            time.sleep(0.01)

    def process_audio_buffer(self):
        if not self.audio_buffer:
            return

        temp_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "temp")
        os.makedirs(temp_dir, exist_ok=True)
        
        audio_filename = os.path.join(temp_dir, f"audio_{uuid.uuid4()}.mp3")
        wav_filename = os.path.join(temp_dir, f"temp_{uuid.uuid4()}.wav")

        try:
            # Сохраняем WAV
            with wave.open(wav_filename, 'wb') as wf:
                wf.setnchannels(self.CHANNELS)
                wf.setsampwidth(self.p.get_sample_size(self.FORMAT))
                wf.setframerate(self.RATE)
                wf.writeframes(b''.join(self.audio_buffer))

            self.audio_buffer = []

            # Конвертируем в MP3 без появления консоли
            subprocess.run([
                'ffmpeg',
                '-i', wav_filename,
                '-c:a', 'libmp3lame',
                '-q:a', '2',
                '-y',
                audio_filename
            ], check=True, capture_output=True, creationflags=subprocess.CREATE_NO_WINDOW)

            # Отправляем на транскрипцию
            self.send_for_transcription(audio_filename)

        except Exception as e:
            self.update_transcript.emit(f"Ошибка обработки: {str(e)}")
        finally:
            if os.path.exists(wav_filename):
                os.remove(wav_filename)
            if os.path.exists(audio_filename):
                os.remove(audio_filename)

    def send_for_transcription(self, audio_filename):
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
                    }
                )

            if response.status_code == 200:
                result = response.json()
                transcribed_text = result.get('text', '').strip()
                if transcribed_text:
                    self.update_transcript.emit(transcribed_text)
                    self.simulate_typing(transcribed_text + " ")
            else:
                self.update_transcript.emit(f"Ошибка API: {response.status_code} - {response.text}")

        except Exception as e:
            self.update_transcript.emit(f"Ошибка транскрипции: {str(e)}")

    def simulate_typing(self, text):
        for char in text:
            self.kbd.type(char)
            time.sleep(0.01)

    def copy_text(self):
        clipboard = QApplication.clipboard()
        clipboard.setText(self.text_edit.toPlainText())
        self.info_label.setText("Текст скопирован")
        QTimer.singleShot(2000, lambda: self.info_label.setText("Готов к записи"))

    @pyqtSlot(str)
    def update_transcript_text(self, text):
        self.text_edit.setText(text)
        self.info_label.setText("Транскрипция завершена")
        QTimer.singleShot(2000, lambda: self.info_label.setText("Готов к записи"))

    def closeEvent(self, event):
        self.program_running = False
        if self.stream:
            self.stream.stop_stream()
            self.stream.close()
        self.p.terminate()
        self.keyboard_listener.stop()
        event.accept()

if __name__ == '__main__':
    app = QApplication(sys.argv)
    app.setStyle('Fusion')
    ex = WhisperGUI()
    ex.show()
    sys.exit(app.exec())