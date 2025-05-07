import sys
from PyQt6.QtWidgets import *
from PyQt6.QtCore import *
from PyQt6.QtGui import *
import pyaudio
# import numpy as np # numpy не используется напрямую в этом коде, можно убрать если не нужен для др. целей
import threading
import time
from pynput import keyboard
from pynput.keyboard import Controller, Key
import warnings
# import requests # УДАЛЕНО
import wave
import os
# import subprocess # subprocess больше не нужен, если не конвертируем в MP3
import uuid
import whisper
import torch
import traceback # Для детального вывода ошибок

warnings.filterwarnings("ignore", category=FutureWarning)

MAX_STORED_AUDIO_FILES = 20
WHISPER_MODEL_NAME = "large-v3" # Используем large-v3

class RecordingIndicator(QWidget):
    def __init__(self):
        super().__init__()
        self._opacity = 1.0
        self.is_recording = False
        self.setFixedSize(24, 24)

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        color = QColor('#FF4444') if self.is_recording else QColor('#555555')
        painter.setBrush(QBrush(color))
        painter.setPen(QPen(QColor('#333333'), 1))
        painter.setOpacity(self.opacity)
        painter.drawEllipse(2, 2, 20, 20)

    @pyqtProperty(float)
    def opacity(self):
        return self._opacity

    @opacity.setter
    def opacity(self, value):
        self._opacity = value
        self.update()

    def set_recording(self, state):
        self.is_recording = state
        if state:
            self.animation = QPropertyAnimation(self, b"opacity", self)
            self.animation.setDuration(1000)
            self.animation.setStartValue(0.4)
            self.animation.setEndValue(1.0)
            self.animation.setLoopCount(-1)
            self.animation.start()
        else:
            if hasattr(self, 'animation') and self.animation and self.animation.state() == QAbstractAnimation.State.Running:
                 self.animation.stop()
            self.opacity = 1.0
        self.update()

class WhisperGUI(QMainWindow):
    update_transcript = pyqtSignal(str)
    signal_update_error = pyqtSignal(str)
    signal_update_info = pyqtSignal(str)
    signal_simulate_typing = pyqtSignal(str)

    def __init__(self):
        super().__init__()
        self.update_transcript.connect(self.update_transcript_text)
        self.signal_update_error.connect(self.update_transcript_error)
        self.signal_update_info.connect(self.update_info_label)
        self.signal_simulate_typing.connect(self.simulate_typing_slot)

        self.whisper_model = None
        self.initUI()
        self.setup_audio()
        self.load_whisper_model()

    def initUI(self):
        self.setWindowTitle(f'Local Whisper Transcriber ({WHISPER_MODEL_NAME})')
        self.setFixedSize(450, 250)
        
        script_dir = os.path.dirname(os.path.abspath(__file__))
        icon_path = os.path.join(script_dir, 'microphone.png')
        if os.path.exists(icon_path):
             self.setWindowIcon(QIcon(icon_path))
        else:
             print(f"Warning: Icon '{icon_path}' not found.")

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

        self.info_label = QLabel('Загрузка...') # Начальное состояние
        self.info_label.setStyleSheet('color: #888888; font-size: 11px;')
        info_layout.addWidget(self.info_label)
        info_layout.addStretch()

        shortcut_label = QLabel('End - выход')
        shortcut_label.setStyleSheet('color: #888888; font-size: 11px;')
        info_layout.addWidget(shortcut_label)

        main_layout.addWidget(info_panel)

    def load_whisper_model(self):
        self.signal_update_info.emit(f"Загрузка модели Whisper '{WHISPER_MODEL_NAME}'...")
        QApplication.processEvents()
        try:
            print(f"Loading Whisper model: {WHISPER_MODEL_NAME}...")
            device = "cuda" if torch.cuda.is_available() else "cpu"
            print(f"Using device: {device}")
            
            self.whisper_model = whisper.load_model(WHISPER_MODEL_NAME, device=device)
            
            self.signal_update_info.emit(f"Модель Whisper '{WHISPER_MODEL_NAME}' загружена ({device}). Готов к записи.")
            print(f"Whisper model '{WHISPER_MODEL_NAME}' loaded successfully on {device}.")
        except Exception as e:
            error_msg = f"Не удалось загрузить модель Whisper '{WHISPER_MODEL_NAME}': {e}\nУбедитесь, что 'openai-whisper' и 'torch' установлены. Для первой загрузки модели нужен интернет (или правильный файл в кеше)."
            print(f"CRITICAL: {error_msg}")
            traceback.print_exc()
            self.signal_update_error.emit(error_msg)
            # QMessageBox.critical(self, "Ошибка загрузки модели Whisper", error_msg) # Может быть слишком навязчиво при старте

    def setup_audio(self):
        self.kbd = Controller()
        self.is_recording = False
        self.audio_buffer = []
        self.program_running = True

        self.CHUNK = 1024
        self.FORMAT = pyaudio.paInt16
        self.CHANNELS = 1
        self.RATE = 16000

        try:
             self.p = pyaudio.PyAudio()
        except Exception as e:
             print(f"FATAL: Could not initialize PyAudio: {e}")
             QMessageBox.critical(self, "Ошибка PyAudio", f"Не удалось инициализировать PyAudio: {e}\nУбедитесь, что PortAudio установлен или доступен.")
             sys.exit(1) 

        self.stream = None

        script_dir = os.path.dirname(os.path.abspath(__file__))
        self.audio_storage_dir = os.path.join(script_dir, "temp_audio")
        try:
            os.makedirs(self.audio_storage_dir, exist_ok=True)
            print(f"Using audio storage directory: {self.audio_storage_dir}")
        except OSError as e:
            critical_msg = f"Не удалось создать/получить доступ к директории для аудио: {self.audio_storage_dir}\n{e}"
            print(f"CRITICAL: {critical_msg}")
            QMessageBox.critical(self, "Ошибка директории", critical_msg)
            sys.exit(1)

        self.keyboard_listener = keyboard.Listener(on_press=self.on_press, daemon=True)
        self.keyboard_listener.start()

        self.record_thread = threading.Thread(target=self.record_audio_loop, daemon=True)
        self.record_thread.start()

    def on_press(self, key):
        try:
            if key == keyboard.Key.insert:
                if self.whisper_model is None:
                    self.signal_update_error.emit("Модель Whisper не загружена. Запись невозможна.")
                    return
                QMetaObject.invokeMethod(self, "toggle_recording", Qt.ConnectionType.QueuedConnection)
            elif key == keyboard.Key.end:
                print("End key pressed, initiating shutdown...")
                self.program_running = False
                QMetaObject.invokeMethod(self, "close", Qt.ConnectionType.QueuedConnection)
                return False 
        except AttributeError:
            pass 
        except Exception as e: 
            print(f"Error in on_press handler: {e}")
            traceback.print_exc()


    @pyqtSlot() 
    def toggle_recording(self):
        if not self.program_running: 
             return
        if self.whisper_model is None:
            self.signal_update_error.emit("Модель Whisper не загружена. Запись невозможна.")
            self.status_label.setText("Ошибка! Модель не загружена.")
            self.info_label.setText("Ошибка загрузки модели")
            return

        self.is_recording = not self.is_recording
        self.indicator.set_recording(self.is_recording) 

        if self.is_recording:
            self.status_label.setText("Идет запись...")
            self.info_label.setText("Записываю...")
            self.audio_buffer = [] 
            if not self.start_audio_stream(): 
                self.is_recording = False
                self.indicator.set_recording(False)
                self.status_label.setText("Ошибка! Не удалось начать запись.")
                self.info_label.setText("Ошибка аудиоустройства")
        else:
            self.status_label.setText("Нажмите Insert для начала записи")
            self.info_label.setText("Обработка записи (локально)...")
            self.stop_audio_stream() 
            audio_data_to_process = b''.join(self.audio_buffer)
            self.audio_buffer = [] 
            if audio_data_to_process:
                 processing_thread = threading.Thread(target=self.process_audio_buffer, args=(audio_data_to_process,), daemon=True)
                 processing_thread.start()
            else:
                 print("No audio data captured, skipping processing.")
                 self.info_label.setText("Запись пуста, обработка пропущена.")
                 QTimer.singleShot(2000, lambda: self.update_info_label("Готов к записи") if self.program_running and self.whisper_model else None)


    def start_audio_stream(self):
         if self.stream is not None:
             print("Stream already exists.")
             return True 

         try:
             self.stream = self.p.open(
                 format=self.FORMAT,
                 channels=self.CHANNELS,
                 rate=self.RATE,
                 input=True,
                 frames_per_buffer=self.CHUNK
             )
             print(f"Audio stream started successfully at {self.RATE} Hz.")
             return True
         except OSError as e:
             error_msg = f"Ошибка открытия аудиопотока (OSError): {e}. Устройство может быть занято."
             print(error_msg)
             self.signal_update_error.emit(error_msg) 
             self.stream = None
             return False
         except Exception as e:
             error_msg = f"Неизвестная ошибка при открытии аудиопотока: {e}"
             print(error_msg)
             traceback.print_exc()
             self.signal_update_error.emit(error_msg)
             self.stream = None
             return False

    def stop_audio_stream(self):
        if self.stream is not None:
            print("Stopping audio stream...")
            try:
                if self.stream.is_active(): # Проверяем, активен ли поток перед остановкой
                    self.stream.stop_stream()
                self.stream.close()
                print("Audio stream stopped and closed.")
            except Exception as e:
                print(f"Error closing audio stream: {e}")
                traceback.print_exc()
            finally:
                 self.stream = None 
        else:
             print("Attempted to stop a non-existent stream.")

    def record_audio_loop(self):
        print("Audio recording loop started.")
        while self.program_running:
            if self.is_recording and self.stream is not None and self.stream.is_active():
                try:
                    data = self.stream.read(self.CHUNK, exception_on_overflow=False)
                    self.audio_buffer.append(data)
                except IOError as e:
                    if e.errno == pyaudio.paInputOverflowed:
                        # print("Input overflowed. Skipping frame.") # Можно раскомментировать для отладки
                        pass
                    elif e.errno == -9988 or "Stream closed" in str(e): # paStreamIsStopped or similar
                         print("Stream closed or stopped during read operation.")
                         time.sleep(0.1) 
                    else:
                        error_msg = f"Audio recording IOError: {e}"
                        print(error_msg)
                        traceback.print_exc()
                        self.signal_update_error.emit(error_msg)
                        QMetaObject.invokeMethod(self, "handle_recording_error", Qt.ConnectionType.QueuedConnection)
                        time.sleep(0.1) 
                except Exception as e:
                    error_msg = f"Unexpected error in recording loop: {e}"
                    print(error_msg)
                    traceback.print_exc()
                    self.signal_update_error.emit(error_msg)
                    QMetaObject.invokeMethod(self, "handle_recording_error", Qt.ConnectionType.QueuedConnection)
                    time.sleep(0.1)
            else:
                time.sleep(0.02) # Небольшая задержка, чтобы не грузить CPU впустую
        print("Audio recording loop finished.")

    @pyqtSlot()
    def handle_recording_error(self):
         print("Handling recording error in main thread.")
         self.stop_audio_stream() # Убедимся, что поток остановлен
         self.is_recording = False
         self.indicator.set_recording(False)
         self.status_label.setText("Ошибка записи!")
         self.info_label.setText("Произошла ошибка аудио")

    def manage_stored_audio_files(self, max_files=MAX_STORED_AUDIO_FILES):
        directory = self.audio_storage_dir
        try:
            files_in_dir = os.listdir(directory)
            audio_wav_files = [f for f in files_in_dir if f.startswith("audio_") and f.endswith(".wav")]
            
            if len(audio_wav_files) > max_files:
                file_details = []
                for f_name in audio_wav_files:
                    f_path = os.path.join(directory, f_name)
                    try:
                        file_details.append((os.path.getmtime(f_path), f_path))
                    except FileNotFoundError:
                        print(f"File not found during management: {f_path}, skipping.")
                        continue
                
                file_details.sort(key=lambda x: x[0]) # Сортировка от старых к новым

                num_to_delete = len(file_details) - max_files
                for i in range(num_to_delete):
                    old_file_path = file_details[i][1]
                    try:
                        os.remove(old_file_path)
                        print(f"Audio limit ({max_files}) reached: Removed old audio file: {os.path.basename(old_file_path)}")
                    except OSError as e:
                        print(f"Error removing old audio file {old_file_path}: {e}")
        except Exception as e:
            print(f"Error managing stored audio files in {directory}: {e}")
            traceback.print_exc()

    def process_audio_buffer(self, audio_data_to_process):
        if not audio_data_to_process:
            print("process_audio_buffer called with empty data.")
            self.signal_update_info.emit("Нет данных для обработки")
            return

        self.signal_update_info.emit("Сохранение аудио...")

        unique_id = uuid.uuid4()
        wav_filename_for_transcription = os.path.join(self.audio_storage_dir, f"audio_{unique_id}.wav")
        
        try:
            print(f"Saving WAV for transcription: {os.path.basename(wav_filename_for_transcription)}")
            with wave.open(wav_filename_for_transcription, 'wb') as wf:
                wf.setnchannels(self.CHANNELS)
                wf.setsampwidth(self.p.get_sample_size(self.FORMAT))
                wf.setframerate(self.RATE)
                wf.writeframes(audio_data_to_process)
            print(f"WAV for transcription saved successfully.")

            self.manage_stored_audio_files()

            self.send_for_transcription(wav_filename_for_transcription)

        except Exception as e:
            error_message = f"Ошибка сохранения WAV: {str(e)}"
            print(error_message)
            traceback.print_exc()
            self.signal_update_error.emit(error_message)
    
    def send_for_transcription(self, audio_filename):
        if not self.whisper_model:
            self.signal_update_error.emit("Локальная модель Whisper не загружена.")
            self.signal_update_info.emit("Модель не загружена")
            print("Error: Whisper model not loaded.")
            return

        if not os.path.exists(audio_filename):
             print(f"Audio file not found for transcription: {audio_filename}")
             self.signal_update_info.emit("Файл аудио не найден")
             return

        self.signal_update_info.emit(f"Локальное распознавание ({WHISPER_MODEL_NAME})...")
        print(f"Sending {os.path.basename(audio_filename)} ({os.path.getsize(audio_filename)} bytes) for local transcription...")
        
        start_time = time.time()
        try:
            use_fp16 = torch.cuda.is_available()
            # Принудительно fp16=False для CPU
            if self.whisper_model.device.type == "cpu": # Более надежная проверка типа устройства
                use_fp16 = False
            
            print(f"Transcribing with: language='ru', fp16={use_fp16}, task='transcribe', device='{self.whisper_model.device.type}'")

            result = self.whisper_model.transcribe(
                audio_filename, 
                language="ru", 
                fp16=use_fp16,
                task="transcribe"
            )
            
            transcribed_text = result.get('text', '').strip()
            end_time = time.time()
            processing_time = end_time - start_time
            print(f"Local transcription took {processing_time:.2f} seconds.")


            if transcribed_text:
                self.update_transcript.emit(transcribed_text)
                self.signal_simulate_typing.emit(transcribed_text + " ")
            else:
                print("Transcription result is empty.")
                self.signal_update_info.emit("Распознан пустой текст (тишина?)")
                QTimer.singleShot(2000, lambda: self.update_info_label("Готов к записи") if self.program_running and self.whisper_model else None)

        except RuntimeError as e:
            error_message = f"Ошибка во время локального распознавания (RuntimeError): {str(e)}"
            print(error_message)
            traceback.print_exc()
            self.signal_update_error.emit(error_message)
            if "CUDA" in str(e) and "out of memory" in str(e).lower():
                 self.signal_update_info.emit("Ошибка CUDA: не хватает памяти. Попробуйте модель поменьше.")
            elif "CUDA" in str(e):
                 self.signal_update_info.emit("Ошибка CUDA. Убедитесь, что драйверы и PyTorch совместимы.")
        except Exception as e:
            error_message = f"Неожиданная ошибка при локальной транскрипции: {str(e)}"
            print(error_message)
            traceback.print_exc()
            self.signal_update_error.emit(error_message)

    @pyqtSlot(str) 
    def simulate_typing_slot(self, text):
         threading.Thread(target=self._simulate_typing_worker, args=(text,), daemon=True).start()

    def _simulate_typing_worker(self, text):
        print(f"Simulating typing: '{text[:50]}...'")
        try:
            for char in text:
                self.kbd.type(char)
                # Небольшая задержка между символами для более естественного ввода
                time.sleep(0.010 + np.random.rand() * 0.010) if 'np' in globals() else time.sleep(0.015) # Добавил проверку np
        except Exception as e:
            print(f"Error during typing simulation: {e}")
            traceback.print_exc()

    @pyqtSlot(str) 
    def update_transcript_text(self, text):
        current_text = self.text_edit.toPlainText()
        new_text = f"{current_text}\n{text}" if current_text else text
        self.text_edit.setText(new_text.strip())
        self.text_edit.moveCursor(QTextCursor.MoveOperation.End)
        self.info_label.setText("Транскрипция добавлена")
        QTimer.singleShot(2000, lambda: self.update_info_label("Готов к записи") if self.program_running and self.whisper_model else None)

    @pyqtSlot(str) 
    def update_transcript_error(self, error_message):
        self.text_edit.append(f"<font color='#FF6B6B'><b>Ошибка:</b> {error_message}</font>")
        self.text_edit.moveCursor(QTextCursor.MoveOperation.End)
        self.info_label.setText("Произошла ошибка")
        QTimer.singleShot(5000, lambda: self.update_info_label("Готов к записи") if self.program_running and self.whisper_model else None)

    @pyqtSlot(str) 
    def update_info_label(self, text):
        if self.program_running:
             if "Загрузка модели Whisper" in self.info_label.text() and "загружена" not in self.info_label.text() and self.whisper_model is None:
                 if "Готов к записи" not in text:
                     self.info_label.setText(text)
             elif self.whisper_model is None and "Готов к записи" in text:
                 pass
             else:
                self.info_label.setText(text)


    def copy_text(self):
        clipboard = QApplication.clipboard()
        clipboard.setText(self.text_edit.toPlainText())
        self.info_label.setText("Текст скопирован")
        QTimer.singleShot(2000, lambda: self.update_info_label("Готов к записи") if self.program_running and self.whisper_model else None)

    def closeEvent(self, event):
        print("Close event received. Shutting down...")
        self.program_running = False 

        if hasattr(self, 'keyboard_listener') and self.keyboard_listener.is_alive():
             print("Stopping keyboard listener (will stop on next event or program exit as daemon)...")
             # pynput listener.stop() может вызывать проблемы из потока, лучше дать ему завершиться как daemon

        self.stop_audio_stream() # Убедимся, что аудиопоток остановлен

        if hasattr(self, 'p') and self.p is not None: # Проверка, что p существует
             print("Terminating PyAudio...")
             try:
                self.p.terminate()
             except Exception as e:
                print(f"Error terminating PyAudio: {e}")
             self.p = None # Явно обнуляем
        
        if self.whisper_model is not None:
            print("Releasing Whisper model from memory...")
            model_device_type = self.whisper_model.device.type
            del self.whisper_model
            self.whisper_model = None # Явно обнуляем
            if model_device_type == "cuda" and torch.cuda.is_available():
                torch.cuda.empty_cache()
            print("Whisper model released.")

        print("Shutdown complete. Accepting close event.")
        event.accept()

if __name__ == '__main__':
    app = QApplication(sys.argv)
    app.setStyle('Fusion') 

    dark_palette = QPalette()
    dark_palette.setColor(QPalette.ColorRole.Window, QColor(30, 30, 30))
    dark_palette.setColor(QPalette.ColorRole.WindowText, Qt.GlobalColor.white)
    dark_palette.setColor(QPalette.ColorRole.Base, QColor(42, 42, 42)) 
    dark_palette.setColor(QPalette.ColorRole.AlternateBase, QColor(66, 66, 66))
    dark_palette.setColor(QPalette.ColorRole.ToolTipBase, Qt.GlobalColor.black) 
    dark_palette.setColor(QPalette.ColorRole.ToolTipText, Qt.GlobalColor.white)
    dark_palette.setColor(QPalette.ColorRole.Text, Qt.GlobalColor.white) 
    dark_palette.setColor(QPalette.ColorRole.Button, QColor(58, 58, 58)) 
    dark_palette.setColor(QPalette.ColorRole.ButtonText, Qt.GlobalColor.white) 
    dark_palette.setColor(QPalette.ColorRole.BrightText, Qt.GlobalColor.red)
    dark_palette.setColor(QPalette.ColorRole.Link, QColor(42, 130, 218))
    dark_palette.setColor(QPalette.ColorRole.Highlight, QColor(65, 65, 65)) 
    dark_palette.setColor(QPalette.ColorRole.HighlightedText, Qt.GlobalColor.white) 
    dark_palette.setColor(QPalette.ColorGroup.Disabled, QPalette.ColorRole.ButtonText, QColor(127, 127, 127))
    dark_palette.setColor(QPalette.ColorGroup.Disabled, QPalette.ColorRole.WindowText, QColor(127, 127, 127))
    dark_palette.setColor(QPalette.ColorGroup.Disabled, QPalette.ColorRole.Text, QColor(127, 127, 127))

    app.setPalette(dark_palette)

    app.setStyleSheet("""
        QWidget { 
            font-family: 'Segoe UI', Arial, sans-serif; 
        }
        QTextEdit {
            background-color: #2A2A2A; 
            color: #E0E0E0;
            border: 1px solid #3A3A3A; 
            border-radius: 4px; 
            padding: 5px;
            selection-background-color: #0078D7; 
            selection-color: white; 
        }
        QScrollBar:vertical {
            border: none;
            background: #2A2A2A;
            width: 10px;
            margin: 0px 0px 0px 0px;
        }
        QScrollBar::handle:vertical {
            background: #505050;
            min-height: 20px;
            border-radius: 5px;
        }
         QScrollBar::handle:vertical:hover {
            background: #606060; 
        }
        QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {
            height: 0px;
            background: none;
        }
        QScrollBar::add-page:vertical, QScrollBar::sub-page:vertical {
            background: none;
        }
        QPushButton {
            background-color: #3A3A3A;
            color: #FFFFFF;
            border: 1px solid #505050;
            border-radius: 4px;
            padding: 5px 10px;
            min-width: 70px; 
            font-size: 12px;
        }
        QPushButton:hover {
            background-color: #4A4A4A; 
            border: 1px solid #606060;
        }
        QPushButton:pressed {
            background-color: #2A2A2A; 
            border: 1px solid #404040;
        }
        QPushButton:disabled { 
             background-color: #2D2D2D;
             color: #6A6A6A;
             border: 1px solid #404040;
        }
        QLabel {
             font-size: 12px; 
        }
        /* Для info_label и shortcut_label можно использовать objectName, если нужно */
        /* self.info_label.setObjectName("infoLabel") */
        /* #infoLabel { color: #888888; font-size: 11px; } */
        QMainWindow {
        }
    """)
    
    # Проверка доступности CUDA при запуске
    print("--- PyTorch CUDA Check ---")
    if torch.cuda.is_available():
        print(f"CUDA is available. PyTorch version: {torch.__version__}")
        print(f"CUDA version by PyTorch: {torch.version.cuda}")
        print(f"Number of GPUs: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            print(f"  GPU {i}: {torch.cuda.get_device_name(i)}")
    else:
        print(f"CUDA is NOT available. PyTorch version: {torch.__version__}")
        print("PyTorch will use CPU. For GPU acceleration, ensure CUDA drivers and a CUDA-enabled PyTorch version are correctly installed.")
    print("--------------------------")

    ex = WhisperGUI()
    ex.show()
    sys.exit(app.exec())