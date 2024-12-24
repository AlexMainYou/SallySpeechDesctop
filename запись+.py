import sys
from PyQt6.QtWidgets import *
from PyQt6.QtCore import *
from PyQt6.QtGui import *
import pyaudio
import threading
import time
from pynput import keyboard
import subprocess
import wave  # <--- Добавлен импорт модуля wave
import os

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

class AudioRecorder(QMainWindow):
    def __init__(self):
        super().__init__()
        self.initUI()
        self.setup_audio()

    def initUI(self):
        self.setWindowTitle('Audio Recorder')
        self.setFixedSize(200, 50)

        central_widget = QWidget()
        self.setCentralWidget(central_widget)

        layout = QHBoxLayout(central_widget)

        self.indicator = RecordingIndicator()
        layout.addWidget(self.indicator)

        self.status_label = QLabel('Нажмите Insert для начала/остановки записи')
        layout.addWidget(self.status_label)

        self.setStyleSheet("""
            QMainWindow {
                background-color: #202020;
            }
            QWidget {
                background-color: #202020;
                color: #ffffff;
            }
            QLabel {
                color: #ffffff;
            }
        """)

    def setup_audio(self):
        self.is_recording = False
        self.audio_buffer = []
        self.program_running = True

        self.CHUNK = 1024
        self.FORMAT = pyaudio.paInt16
        self.CHANNELS = 1
        self.RATE = 48000  # Увеличиваем частоту дискретизации для лучшего качества

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
            self.save_audio()

    def save_audio(self):
        raw_filename = "recorded_audio_raw.wav"
        output_filename = "recorded_audio.mp3"

        # Сохраняем сырой PCM в WAV для последующего сжатия
        with wave.open(raw_filename, 'wb') as wf:
            wf.setnchannels(self.CHANNELS)
            wf.setsampwidth(self.p.get_sample_size(self.FORMAT))
            wf.setframerate(self.RATE)
            wf.writeframes(b''.join(self.audio_buffer))

        self.audio_buffer = []

        # Сжимаем WAV в MP3 с помощью FFmpeg
        try:
            # Используем LAME для кодирования в MP3, VBR для лучшего соотношения качество/размер
            subprocess.run([
                'ffmpeg',
                '-y',  # Перезаписывать выходной файл без запроса
                '-f', 's16le',  # Формат входных данных
                '-ar', str(self.RATE),  # Частота дискретизации
                '-ac', str(self.CHANNELS),  # Количество каналов
                '-i', raw_filename,  # Входной файл
                '-c:a', 'libmp3lame',  # Использовать кодек LAME
                '-q:a', '2',  # Качество VBR, 0 (лучшее) - 9 (худшее), 2 - хороший компромисс
                output_filename  # Выходной файл
            ], check=True)
            print(f"Аудио сохранено в {output_filename}")
        except subprocess.CalledProcessError as e:
            print(f"Ошибка при сжатии аудио: {e}")
        finally:
            os.remove(raw_filename)  # Удаляем временный WAV файл

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

    def closeEvent(self, event):
        self.program_running = False
        self.keyboard_listener.stop()
        self.p.terminate()
        event.accept()

if __name__ == '__main__':
    app = QApplication(sys.argv)
    ex = AudioRecorder()
    ex.show()
    sys.exit(app.exec())