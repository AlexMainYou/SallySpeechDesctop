import sys
from PyQt6.QtWidgets import *
from PyQt6.QtCore import *
from PyQt6.QtGui import *
import pyaudio
import numpy as np
import threading
import time
from pynput import keyboard
import warnings
import wave
import os
import subprocess
import uuid
import ctypes
from ctypes import wintypes
from groq import APIConnectionError, APIStatusError, APITimeoutError, Groq
# tempfile не используется для основной директории аудио, но может быть полезен для других временных нужд

warnings.filterwarnings("ignore", category=FutureWarning)

def load_env_value(name, default=None):
    app_dir = os.path.dirname(sys.executable) if getattr(sys, "frozen", False) else os.path.dirname(os.path.abspath(__file__))
    env_path = os.path.join(app_dir, ".env")
    try:
        with open(env_path, "r", encoding="utf-8") as env_file:
            for raw_line in env_file:
                line = raw_line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                key, value = line.split("=", 1)
                if key.strip() == name:
                    return value.strip().strip('"').strip("'")
    except FileNotFoundError:
        pass
    except OSError as e:
        print(f"Warning: could not read .env file: {e}")
    return os.environ.get(name, default)

GROQ_API_KEY = load_env_value("GROQ_API_KEY")
GROQ_MODEL = "whisper-large-v3"
TOGGLE_HOTKEY_LABEL = "Ctrl+Alt+Space"
TRANSCRIPTION_AUDIO_RATE = 16000
TRANSCRIPTION_AUDIO_BITRATE = "64k"

MAX_STORED_AUDIO_FILES = 20

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

        self.initUI()
        self.setup_audio()


    def initUI(self):
        self.setWindowTitle('Whisper Transcriber (whisper-v3)')
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

        self.status_label = QLabel(f'Нажмите {TOGGLE_HOTKEY_LABEL} для начала записи')
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

        shortcut_label = QLabel(f'{TOGGLE_HOTKEY_LABEL} - запись, End - выход')
        shortcut_label.setStyleSheet('color: #888888; font-size: 11px;')
        info_layout.addWidget(shortcut_label)

        main_layout.addWidget(info_panel)

    def setup_audio(self):
        self.is_recording = False
        self.audio_buffer = []
        self.program_running = True
        self.pressed_keys = set()
        self.toggle_hotkey_active = False
        self.groq_client = Groq(api_key=GROQ_API_KEY, timeout=90.0) if GROQ_API_KEY else None

        self.CHUNK = 1024
        self.FORMAT = pyaudio.paInt16
        self.CHANNELS = 1
        self.RATE = 44100 

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

        self.keyboard_listener = keyboard.Listener(on_press=self.on_press, on_release=self.on_release, daemon=True)
        self.keyboard_listener.start()

        self.record_thread = threading.Thread(target=self.record_audio_loop, daemon=True)
        self.record_thread.start()

    def on_press(self, key):
        try:
            self.pressed_keys.add(key)

            if key == keyboard.Key.end:
                print("End key pressed, initiating shutdown...")
                self.program_running = False
                QMetaObject.invokeMethod(self, "close", Qt.ConnectionType.QueuedConnection)
                return False 

            if self.is_toggle_hotkey_pressed() and not self.toggle_hotkey_active:
                self.toggle_hotkey_active = True
                QMetaObject.invokeMethod(self, "toggle_recording", Qt.ConnectionType.QueuedConnection)
        except AttributeError:
            pass 
        except Exception as e: 
            print(f"Error in on_press handler: {e}")

    def on_release(self, key):
        try:
            self.pressed_keys.discard(key)
            if not self.is_toggle_hotkey_pressed():
                self.toggle_hotkey_active = False
        except Exception as e:
            print(f"Error in on_release handler: {e}")

    def is_toggle_hotkey_pressed(self):
        ctrl_keys = {keyboard.Key.ctrl_l, keyboard.Key.ctrl_r, getattr(keyboard.Key, "ctrl", keyboard.Key.ctrl_l)}
        alt_keys = {
            keyboard.Key.alt_l,
            keyboard.Key.alt_r,
            getattr(keyboard.Key, "alt", keyboard.Key.alt_l),
            getattr(keyboard.Key, "alt_gr", keyboard.Key.alt_r),
        }
        return (
            keyboard.Key.space in self.pressed_keys
            and bool(ctrl_keys & self.pressed_keys)
            and bool(alt_keys & self.pressed_keys)
        )

    @pyqtSlot() 
    def toggle_recording(self):
        if not self.program_running: 
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
            self.status_label.setText(f"Нажмите {TOGGLE_HOTKEY_LABEL} для начала записи")
            self.info_label.setText("Обработка записи...")
            self.stop_audio_stream() 
            audio_data_to_process = b''.join(self.audio_buffer)
            self.audio_buffer = [] 
            if audio_data_to_process:
                 processing_thread = threading.Thread(target=self.process_audio_buffer, args=(audio_data_to_process,), daemon=True)
                 processing_thread.start()
            else:
                 print("No audio data captured, skipping processing.")
                 self.info_label.setText("Запись пуста, обработка пропущена.")
                 QTimer.singleShot(2000, lambda: self.update_info_label("Готов к записи") if self.program_running else None)

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
             print("Audio stream started successfully.")
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
             self.signal_update_error.emit(error_msg)
             self.stream = None
             return False

    def stop_audio_stream(self):
        if self.stream is not None:
            print("Stopping audio stream...")
            try:
                if self.stream.is_active():
                    self.stream.stop_stream()
                self.stream.close()
                print("Audio stream stopped and closed.")
            except Exception as e:
                print(f"Error closing audio stream: {e}")
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
                        pass
                    elif e.errno == -9988 or "Stream closed" in str(e):
                         print("Stream closed during read operation.")
                         time.sleep(0.1) 
                    else:
                        error_msg = f"Audio recording IOError: {e}"
                        print(error_msg)
                        self.signal_update_error.emit(error_msg)
                        QMetaObject.invokeMethod(self, "handle_recording_error", Qt.ConnectionType.QueuedConnection)
                        time.sleep(0.1) 
                except Exception as e:
                    error_msg = f"Unexpected error in recording loop: {e}"
                    print(error_msg)
                    self.signal_update_error.emit(error_msg)
                    QMetaObject.invokeMethod(self, "handle_recording_error", Qt.ConnectionType.QueuedConnection)
                    time.sleep(0.1)
            else:
                time.sleep(0.02)
        print("Audio recording loop finished.")

    @pyqtSlot()
    def handle_recording_error(self):
         print("Handling recording error in main thread.")
         self.stop_audio_stream()
         self.is_recording = False
         self.indicator.set_recording(False)
         self.status_label.setText("Ошибка записи!")
         self.info_label.setText("Произошла ошибка аудио")

    def manage_stored_audio_files(self, max_files=MAX_STORED_AUDIO_FILES):
        directory = self.audio_storage_dir
        try:
            files_in_dir = os.listdir(directory)
            audio_mp3_files = [f for f in files_in_dir if f.startswith("audio_") and f.endswith(".mp3")]
            
            if len(audio_mp3_files) > max_files:
                file_details = []
                for f_name in audio_mp3_files:
                    f_path = os.path.join(directory, f_name)
                    try:
                        file_details.append((os.path.getmtime(f_path), f_path))
                    except FileNotFoundError:
                        print(f"File not found during management: {f_path}, skipping.")
                        continue
                
                file_details.sort(key=lambda x: x[0])

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

    def process_audio_buffer(self, audio_data_to_process):
        if not audio_data_to_process:
            print("process_audio_buffer called with empty data.")
            self.signal_update_info.emit("Нет данных для обработки")
            return

        self.signal_update_info.emit("Сохранение и конвертация...")

        unique_id = uuid.uuid4()
        wav_filename = os.path.join(self.audio_storage_dir, f"temp_{unique_id}.wav")
        mp3_to_store_and_send = os.path.join(self.audio_storage_dir, f"audio_{unique_id}.mp3")

        try:
            print(f"Saving WAV: {os.path.basename(wav_filename)}")
            with wave.open(wav_filename, 'wb') as wf:
                wf.setnchannels(self.CHANNELS)
                wf.setsampwidth(self.p.get_sample_size(self.FORMAT))
                wf.setframerate(self.RATE)
                wf.writeframes(audio_data_to_process)
            print(f"WAV saved successfully.")

            print(f"Starting conversion to MP3: {os.path.basename(mp3_to_store_and_send)}")
            startupinfo = None
            if os.name == 'nt': 
                startupinfo = subprocess.STARTUPINFO()
                startupinfo.dwFlags |= subprocess.STARTF_USESHOWWINDOW
                startupinfo.wShowWindow = subprocess.SW_HIDE

            ffmpeg_command = [
                'ffmpeg', '-y', 
                '-i', wav_filename,
                '-vn', 
                '-acodec', 'libmp3lame',
                '-ab', TRANSCRIPTION_AUDIO_BITRATE, 
                '-ar', str(TRANSCRIPTION_AUDIO_RATE),
                '-ac', str(self.CHANNELS),
                mp3_to_store_and_send
            ]
            
            conversion_started_at = time.perf_counter()
            process = subprocess.run(
                ffmpeg_command,
                check=True, 
                capture_output=True, 
                text=True, 
                encoding='utf-8', 
                startupinfo=startupinfo
            )
            
            if process.returncode == 0:
                 conversion_elapsed = time.perf_counter() - conversion_started_at
                 print(f"MP3 conversion successful: {os.path.basename(mp3_to_store_and_send)} in {conversion_elapsed:.2f}s")
                 self.manage_stored_audio_files() 
            else:
                 raise subprocess.CalledProcessError(process.returncode, ffmpeg_command, output=process.stdout, stderr=process.stderr)

            # Передаем созданный MP3 файл в функцию транскрипции
            self.send_for_transcription(mp3_to_store_and_send) # <--- Имя файла MP3

        except FileNotFoundError:
             error_message = "Ошибка: ffmpeg не найден. Установите ffmpeg и добавьте его в PATH."
             print(error_message)
             self.signal_update_error.emit(error_message)
        except subprocess.CalledProcessError as e:
            error_details = f"stderr: {e.stderr}\nstdout: {e.stdout}"
            error_message = f"Ошибка конвертации ffmpeg (код {e.returncode}). См. консоль."
            print(f"{error_message}\n{error_details}")
            self.signal_update_error.emit(f"Ошибка ffmpeg: {e.stderr[:200]}...") 
        except Exception as e:
            error_message = f"Ошибка обработки аудио: {str(e)}"
            print(error_message)
            self.signal_update_error.emit(error_message)
        finally:
            try:
                if os.path.exists(wav_filename):
                    os.remove(wav_filename)
            except OSError as e_rem:
                print(f"Warning: Could not remove temp WAV file {os.path.basename(wav_filename)}: {e_rem}")
    
    # --- Эта функция приведена к виду, максимально близкому к вашему исходному ---
    def send_for_transcription(self, audio_filename): # Используем audio_filename как имя параметра
        """Отправляет аудиофайл в API и обрабатывает ответ."""
        if not os.path.exists(audio_filename):
             print(f"Audio file not found for transcription: {audio_filename}")
             self.signal_update_info.emit("Файл аудио не найден")
             return

        self.signal_update_info.emit("Отправка на распознавание...")
        print(f"Sending {os.path.basename(audio_filename)} ({os.path.getsize(audio_filename)} bytes) for transcription...")

        try:
            if self.groq_client is None:
                error_message = "Ошибка API: переменная окружения GROQ_API_KEY не задана."
                print(error_message)
                self.signal_update_error.emit(error_message)
                return

            with open(audio_filename, 'rb') as f:
                transcription_started_at = time.perf_counter()
                transcription = self.groq_client.audio.transcriptions.create(
                    file=(os.path.basename(audio_filename), f.read()),
                    model=GROQ_MODEL,
                    temperature=0,
                    response_format="verbose_json",
                    language="ru",
                )
                transcription_elapsed = time.perf_counter() - transcription_started_at
                print(f"Groq transcription completed in {transcription_elapsed:.2f}s")

            transcribed_text = (getattr(transcription, "text", "") or "").strip()

            if transcribed_text:
                self.update_transcript.emit(transcribed_text)
                self.signal_simulate_typing.emit(transcribed_text + " ")
            else:
                print("Transcription result is empty.")
                self.signal_update_info.emit("Распознан пустой текст (тишина?)")
                QTimer.singleShot(2000, lambda: self.update_info_label("Готов к записи") if self.program_running else None)

        except APITimeoutError:
             error_message = "Ошибка API: Превышен таймаут ожидания ответа."
             print(error_message)
             self.signal_update_error.emit(error_message)
        except APIStatusError as e:
            error_message = f"Ошибка API ({e.status_code}): {e.response.text}"
            print(error_message)
            self.signal_update_error.emit(error_message)
        except APIConnectionError as e:
             error_message = f"Ошибка сети при запросе к API: {str(e)}"
             print(error_message)
             self.signal_update_error.emit(error_message)
        except Exception as e:
            error_message = f"Неожиданная ошибка при обработке ответа API: {str(e)}"
            print(error_message)
            self.signal_update_error.emit(error_message)

    @pyqtSlot(str) 
    def simulate_typing_slot(self, text):
         threading.Thread(target=self._insert_text_worker, args=(text,), daemon=True).start()

    def _insert_text_worker(self, text):
        print(f"Inserting transcription with Unicode input: '{text[:50]}...'")
        try:
            time.sleep(0.05)
            self._send_unicode_text(text)
        except Exception as e:
            print(f"Error during Unicode text insertion: {e}")

    def _send_unicode_text(self, text):
        if os.name != 'nt':
            raise RuntimeError("Unicode text insertion without clipboard is only implemented on Windows.")

        input_keyboard = 1
        keyeventf_keyup = 0x0002
        keyeventf_unicode = 0x0004
        user32 = ctypes.WinDLL("user32", use_last_error=True)

        class KEYBDINPUT(ctypes.Structure):
            _fields_ = [
                ("wVk", wintypes.WORD),
                ("wScan", wintypes.WORD),
                ("dwFlags", wintypes.DWORD),
                ("time", wintypes.DWORD),
                ("dwExtraInfo", ctypes.c_size_t),
            ]

        class MOUSEINPUT(ctypes.Structure):
            _fields_ = [
                ("dx", wintypes.LONG),
                ("dy", wintypes.LONG),
                ("mouseData", wintypes.DWORD),
                ("dwFlags", wintypes.DWORD),
                ("time", wintypes.DWORD),
                ("dwExtraInfo", ctypes.c_size_t),
            ]

        class HARDWAREINPUT(ctypes.Structure):
            _fields_ = [
                ("uMsg", wintypes.DWORD),
                ("wParamL", wintypes.WORD),
                ("wParamH", wintypes.WORD),
            ]

        class INPUT_UNION(ctypes.Union):
            _fields_ = [
                ("mi", MOUSEINPUT),
                ("ki", KEYBDINPUT),
                ("hi", HARDWAREINPUT),
            ]

        class INPUT(ctypes.Structure):
            _anonymous_ = ("u",)
            _fields_ = [("type", wintypes.DWORD), ("u", INPUT_UNION)]

        user32.SendInput.argtypes = (wintypes.UINT, ctypes.POINTER(INPUT), ctypes.c_int)
        user32.SendInput.restype = wintypes.UINT

        def make_input(code_unit, key_up=False):
            flags = keyeventf_unicode | (keyeventf_keyup if key_up else 0)
            return INPUT(
                type=input_keyboard,
                ki=KEYBDINPUT(
                    wVk=0,
                    wScan=code_unit,
                    dwFlags=flags,
                    time=0,
                    dwExtraInfo=0,
                ),
            )

        utf16 = text.encode("utf-16-le")
        code_units = [
            int.from_bytes(utf16[i:i + 2], "little")
            for i in range(0, len(utf16), 2)
        ]

        for start in range(0, len(code_units), 64):
            events = []
            for code_unit in code_units[start:start + 64]:
                events.append(make_input(code_unit))
                events.append(make_input(code_unit, key_up=True))

            inputs = (INPUT * len(events))(*events)
            sent = user32.SendInput(len(events), inputs, ctypes.sizeof(INPUT))
            if sent != len(events):
                raise ctypes.WinError(ctypes.get_last_error())

    @pyqtSlot(str) 
    def update_transcript_text(self, text):
        current_text = self.text_edit.toPlainText()
        new_text = f"{current_text}\n{text}" if current_text else text
        self.text_edit.setText(new_text.strip())
        self.text_edit.moveCursor(QTextCursor.MoveOperation.End)
        self.info_label.setText("Транскрипция добавлена")
        QTimer.singleShot(2000, lambda: self.update_info_label("Готов к записи") if self.program_running else None)

    @pyqtSlot(str) 
    def update_transcript_error(self, error_message):
        self.text_edit.append(f"<font color='#FF6B6B'><b>Ошибка:</b> {error_message}</font>")
        self.text_edit.moveCursor(QTextCursor.MoveOperation.End)
        self.info_label.setText("Произошла ошибка")
        QTimer.singleShot(5000, lambda: self.update_info_label("Готов к записи") if self.program_running else None)

    @pyqtSlot(str) 
    def update_info_label(self, text):
        if self.program_running: 
             self.info_label.setText(text)

    def copy_text(self):
        clipboard = QApplication.clipboard()
        clipboard.setText(self.text_edit.toPlainText())
        self.info_label.setText("Текст скопирован")
        QTimer.singleShot(2000, lambda: self.update_info_label("Готов к записи") if self.program_running else None)

    def closeEvent(self, event):
        print("Close event received. Shutting down...")
        self.program_running = False 

        if hasattr(self, 'keyboard_listener') and self.keyboard_listener.is_alive():
             print("Stopping keyboard listener (will stop on next event or program exit as daemon)...")

        self.stop_audio_stream()

        if hasattr(self, 'p'):
             print("Terminating PyAudio...")
             self.p.terminate()
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
        #info_label QLabel, #shortcut_label QLabel { 
            color: #888888;
            font-size: 11px;
        }
        QMainWindow {
        }
    """)

    ex = WhisperGUI()
    ex.show()
    sys.exit(app.exec())
