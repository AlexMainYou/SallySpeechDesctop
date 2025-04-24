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
import tempfile # Используем для временной папки

warnings.filterwarnings("ignore", category=FutureWarning)

# !!! ВАЖНО: Замените на ваш реальный API ключ Fireworks AI !!!
# Не храните ключ прямо в коде для продакшн приложений. Используйте переменные окружения.
FIREWORKS_API_KEY = "YOUR_FIREWORKS_API_KEY" # <--- ЗАМЕНИТЕ ЭТОТ КЛЮЧ

class RecordingIndicator(QWidget):
    def __init__(self):
        super().__init__()
        self._opacity = 1.0 # Инициализируем приватное поле
        self.is_recording = False
        self.setFixedSize(24, 24)
        # self.animation создается при запуске set_recording

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        color = QColor('#FF4444') if self.is_recording else QColor('#555555')
        painter.setBrush(QBrush(color))
        painter.setPen(QPen(QColor('#333333'), 1))
        # Используем значение из property
        painter.setOpacity(self.opacity)
        painter.drawEllipse(2, 2, 20, 20)

    # Используем property для opacity, чтобы анимация работала
    @pyqtProperty(float)
    def opacity(self):
        return self._opacity

    @opacity.setter
    def opacity(self, value):
        # print(f"Setting opacity: {value}") # Для дебага анимации
        self._opacity = value
        self.update() # Перерисовываем виджет при изменении прозрачности

    def set_recording(self, state):
        self.is_recording = state
        if state:
            # Создаем анимацию, если еще не создана или нужна новая
            self.animation = QPropertyAnimation(self, b"opacity", self) # Указываем родителя
            self.animation.setDuration(1000)
            self.animation.setStartValue(0.4)
            self.animation.setEndValue(1.0)
            self.animation.setLoopCount(-1) # Бесконечный цикл
            self.animation.start()
        else:
            # Останавливаем анимацию, если она существует и запущена
            if hasattr(self, 'animation') and self.animation and self.animation.state() == QAbstractAnimation.State.Running:
                 self.animation.stop()
            self.opacity = 1.0 # Возвращаем полную непрозрачность явно через сеттер
        self.update() # Обновляем виджет на всякий случай

class WhisperGUI(QMainWindow):
    update_transcript = pyqtSignal(str)
    # Сигнал для передачи ошибок в основной поток
    signal_update_error = pyqtSignal(str)
    # Сигнал для обновления статусных сообщений
    signal_update_info = pyqtSignal(str)
    # Сигнал для имитации ввода
    signal_simulate_typing = pyqtSignal(str)

    def __init__(self):
        super().__init__()
        # Подключаем сигналы к слотам
        self.update_transcript.connect(self.update_transcript_text)
        self.signal_update_error.connect(self.update_transcript_error)
        self.signal_update_info.connect(self.update_info_label)
        self.signal_simulate_typing.connect(self.simulate_typing_slot)

        self.initUI()
        self.setup_audio()


    def initUI(self):
        # Используем название модели из API
        self.setWindowTitle('Whisper Transcriber (whisper-v3)')
        self.setFixedSize(450, 250)
        # Попробуем найти иконку, если нет - не страшно
        icon_path = 'microphone.png'
        if os.path.exists(icon_path):
             self.setWindowIcon(QIcon(icon_path))
        # else:
             # print(f"Warning: Icon '{icon_path}' not found.") # Уже выводится


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

        # Стили применяются при инициализации app

    def setup_audio(self):
        self.kbd = Controller()
        self.is_recording = False
        self.audio_buffer = []
        self.program_running = True

        self.CHUNK = 1024
        self.FORMAT = pyaudio.paInt16
        self.CHANNELS = 1
        self.RATE = 44100 # Стандартная частота дискретизации

        try:
             self.p = pyaudio.PyAudio()
        except Exception as e:
             print(f"FATAL: Could not initialize PyAudio: {e}")
             QMessageBox.critical(self, "Ошибка PyAudio", f"Не удалось инициализировать PyAudio: {e}\nУбедитесь, что PortAudio установлен или доступен.")
             sys.exit(1) # Выход, если аудио не работает

        self.stream = None

        # Создаем временную директорию при запуске
        self.temp_dir = os.path.join(tempfile.gettempdir(), "whisper_gui_temp")
        try:
            os.makedirs(self.temp_dir, exist_ok=True)
            print(f"Using temporary directory: {self.temp_dir}")
        except OSError as e:
            print(f"Warning: Could not create temporary directory {self.temp_dir}: {e}")
            # Попробуем использовать директорию скрипта как запасной вариант
            script_dir = os.path.dirname(os.path.abspath(__file__))
            self.temp_dir = os.path.join(script_dir, "temp_audio")
            os.makedirs(self.temp_dir, exist_ok=True)
            print(f"Using fallback temporary directory: {self.temp_dir}")


        # Запускаем слушатель клавиатуры как демон
        self.keyboard_listener = keyboard.Listener(on_press=self.on_press, daemon=True)
        self.keyboard_listener.start()

        # Запускаем поток для цикла чтения аудио
        self.record_thread = threading.Thread(target=self.record_audio_loop, daemon=True)
        self.record_thread.start()


    def on_press(self, key):
        try:
            if key == keyboard.Key.insert:
                # Используем QueuedConnection для вызова слота из другого потока
                QMetaObject.invokeMethod(self, "toggle_recording", Qt.ConnectionType.QueuedConnection)
            elif key == keyboard.Key.end:
                print("End key pressed, initiating shutdown...")
                self.program_running = False
                # Закрытие окна также через основной поток
                QMetaObject.invokeMethod(self, "close", Qt.ConnectionType.QueuedConnection)
                return False # Останавливаем слушатель
        except AttributeError:
            pass # Игнорируем клавиши без атрибута `name` или `char`
        except Exception as e: # Ловим другие возможные ошибки
            print(f"Error in on_press handler: {e}")


    @pyqtSlot() # Явно помечаем как слот Qt
    def toggle_recording(self):
        if not self.program_running: # Не переключать, если уже закрываемся
             return

        self.is_recording = not self.is_recording
        self.indicator.set_recording(self.is_recording) # Обновляем индикатор

        if self.is_recording:
            self.status_label.setText("Идет запись...")
            self.info_label.setText("Записываю...")
            self.audio_buffer = [] # Очищаем буфер перед новой записью
            if not self.start_audio_stream(): # Пытаемся запустить поток
                # Если не удалось запустить, сбрасываем состояние
                self.is_recording = False
                self.indicator.set_recording(False)
                self.status_label.setText("Ошибка! Не удалось начать запись.")
                self.info_label.setText("Ошибка аудиоустройства")
        else:
            self.status_label.setText("Нажмите Insert для начала записи")
            self.info_label.setText("Обработка записи...")
            self.stop_audio_stream() # Останавливаем поток аудио
            # Копируем данные буфера перед запуском потока обработки
            audio_data_to_process = b''.join(self.audio_buffer)
            self.audio_buffer = [] # Очищаем основной буфер
            if audio_data_to_process:
                 # Запускаем обработку в отдельном потоке
                 processing_thread = threading.Thread(target=self.process_audio_buffer, args=(audio_data_to_process,), daemon=True)
                 processing_thread.start()
            else:
                 print("No audio data captured, skipping processing.")
                 self.info_label.setText("Запись пуста, обработка пропущена.")
                 QTimer.singleShot(2000, lambda: self.update_info_label("Готов к записи") if self.program_running else None)



    def start_audio_stream(self):
         """Пытается открыть аудиопоток. Возвращает True в случае успеха, False при ошибке."""
         if self.stream is not None:
             print("Stream already exists.")
             return True # Поток уже открыт

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
             # Часто ошибка OSError возникает, если устройство занято или недоступно
             error_msg = f"Ошибка открытия аудиопотока (OSError): {e}. Устройство может быть занято."
             print(error_msg)
             self.signal_update_error.emit(error_msg) # Сообщаем об ошибке в GUI
             self.stream = None
             return False
         except Exception as e:
             error_msg = f"Неизвестная ошибка при открытии аудиопотока: {e}"
             print(error_msg)
             self.signal_update_error.emit(error_msg)
             self.stream = None
             return False

    def stop_audio_stream(self):
        """Безопасно останавливает и закрывает аудиопоток."""
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
                 self.stream = None # Гарантированно сбрасываем ссылку
        else:
             print("Attempted to stop a non-existent stream.")


    def record_audio_loop(self):
        """Цикл чтения данных из аудиопотока в отдельном потоке."""
        print("Audio recording loop started.")
        while self.program_running:
            if self.is_recording and self.stream is not None and self.stream.is_active():
                try:
                    data = self.stream.read(self.CHUNK, exception_on_overflow=False)
                    # Добавляем данные в буфер (взаимодействие с общим ресурсом, но append потокобезопасен для GIL)
                    self.audio_buffer.append(data)
                except IOError as e:
                    # input overflowed - частое явление, можно игнорировать или логировать реже
                    if e.errno == pyaudio.paInputOverflowed:
                        # print("Warning: Audio input overflowed.")
                        pass
                    # Stream closed - возникает при остановке потока из другого места
                    elif e.errno == -9988 or "Stream closed" in str(e):
                         print("Stream closed during read operation.")
                         # Не нужно вызывать stop_audio_stream здесь, он вызовется из toggle_recording
                         time.sleep(0.1) # Даем время на закрытие
                    else:
                        error_msg = f"Audio recording IOError: {e}"
                        print(error_msg)
                        # Сообщаем об ошибке в GUI через сигнал
                        self.signal_update_error.emit(error_msg)
                        # Пытаемся корректно остановить поток и сбросить запись
                        QMetaObject.invokeMethod(self, "handle_recording_error", Qt.ConnectionType.QueuedConnection)
                        time.sleep(0.1) # Пауза после ошибки
                except Exception as e:
                    error_msg = f"Unexpected error in recording loop: {e}"
                    print(error_msg)
                    self.signal_update_error.emit(error_msg)
                    QMetaObject.invokeMethod(self, "handle_recording_error", Qt.ConnectionType.QueuedConnection)
                    time.sleep(0.1)
            else:
                # Спим, если не записываем или поток не активен
                time.sleep(0.02)
        print("Audio recording loop finished.")

    @pyqtSlot()
    def handle_recording_error(self):
         """Обработчик для сброса состояния записи при ошибке в потоке чтения."""
         print("Handling recording error in main thread.")
         self.stop_audio_stream()
         self.is_recording = False
         self.indicator.set_recording(False)
         self.status_label.setText("Ошибка записи!")
         self.info_label.setText("Произошла ошибка аудио")


    def process_audio_buffer(self, audio_data_to_process):
        """Обрабатывает переданные аудиоданные: сохраняет, конвертирует, отправляет."""
        if not audio_data_to_process:
            print("process_audio_buffer called with empty data.")
            self.signal_update_info.emit("Нет данных для обработки")
            return

        # Обновляем статус в GUI через сигнал
        self.signal_update_info.emit("Сохранение и конвертация...")

        unique_id = uuid.uuid4()
        wav_filename = os.path.join(self.temp_dir, f"temp_{unique_id}.wav")
        # Используем MP3 для отправки
        audio_filename = os.path.join(self.temp_dir, f"audio_{unique_id}.mp3")

        try:
            # 1. Сохраняем WAV
            print(f"Saving WAV: {wav_filename}")
            with wave.open(wav_filename, 'wb') as wf:
                wf.setnchannels(self.CHANNELS)
                wf.setsampwidth(self.p.get_sample_size(self.FORMAT))
                wf.setframerate(self.RATE)
                wf.writeframes(audio_data_to_process)
            print(f"WAV saved successfully.")

            # 2. Конвертируем в MP3 с помощью ffmpeg
            print(f"Starting conversion to MP3: {audio_filename}")
            startupinfo = None
            if os.name == 'nt': # Скрытие окна консоли в Windows
                startupinfo = subprocess.STARTUPINFO()
                startupinfo.dwFlags |= subprocess.STARTF_USESHOWWINDOW
                startupinfo.wShowWindow = subprocess.SW_HIDE

            ffmpeg_command = [
                'ffmpeg', '-y', # Перезаписывать без запроса
                '-i', wav_filename,
                '-vn', # Без видео
                '-acodec', 'libmp3lame',
                '-ab', '192k', # Битрейт
                '-ar', str(self.RATE),
                '-ac', str(self.CHANNELS),
                audio_filename
            ]
            # print(f"Running ffmpeg: {' '.join(ffmpeg_command)}") # Дебаг команды

            process = subprocess.run(
                ffmpeg_command,
                check=True, # Выбросить исключение при ошибке ffmpeg
                capture_output=True, # Захватить вывод
                text=True, # Текстовый вывод
                encoding='utf-8', # Указать кодировку вывода ffmpeg
                startupinfo=startupinfo
            )
            # Выводим stderr, т.к. ffmpeg часто пишет туда информацию о процессе
            # print("FFmpeg stderr:", process.stderr)
            # print("FFmpeg stdout:", process.stdout)
            if process.returncode == 0:
                 print(f"MP3 conversion successful: {audio_filename}")
            else:
                 # Эта ветка не должна выполниться из-за check=True, но на всякий случай
                 raise subprocess.CalledProcessError(process.returncode, ffmpeg_command, output=process.stdout, stderr=process.stderr)


            # 3. Отправляем MP3 на транскрипцию
            self.send_for_transcription(audio_filename)

        except FileNotFoundError:
             error_message = "Ошибка: ffmpeg не найден. Установите ffmpeg и добавьте его в PATH."
             print(error_message)
             self.signal_update_error.emit(error_message)
        except subprocess.CalledProcessError as e:
            # Логгируем детали ошибки ffmpeg
            error_details = f"stderr: {e.stderr}\nstdout: {e.stdout}"
            error_message = f"Ошибка конвертации ffmpeg (код {e.returncode}). См. консоль."
            print(f"{error_message}\n{error_details}")
            self.signal_update_error.emit(f"Ошибка ffmpeg: {e.stderr[:200]}...") # Показываем часть ошибки
        except Exception as e:
            error_message = f"Ошибка обработки аудио: {str(e)}"
            print(error_message)
            # Передаем ошибку в основной поток через сигнал
            self.signal_update_error.emit(error_message)
        finally:
            # Удаляем временные файлы в любом случае
            for f in [wav_filename, audio_filename]:
                 try:
                     if os.path.exists(f):
                         os.remove(f)
                         # print(f"Removed temp file: {f}")
                 except OSError as e_rem:
                      print(f"Warning: Could not remove temp file {f}: {e_rem}")


    def send_for_transcription(self, audio_filename):
        """Отправляет аудиофайл в API и обрабатывает ответ."""
        if not os.path.exists(audio_filename):
             print(f"Audio file not found for transcription: {audio_filename}")
             self.signal_update_info.emit("Файл аудио не найден")
             return

        self.signal_update_info.emit("Отправка на распознавание...")
        print(f"Sending {os.path.basename(audio_filename)} ({os.path.getsize(audio_filename)} bytes) for transcription...")

        try:
            with open(audio_filename, 'rb') as f:
                files_data = {"file": (os.path.basename(audio_filename), f, 'audio/mpeg')}
                payload_data = {
                     # --- ИЗМЕНЕНИЕ ЗДЕСЬ ---
                     "model": "whisper-v3", # Возвращаем имя, которое ожидает API
                     # -----------------------
                     "temperature": "0.2",
                     "vad_model": "silero", # Используем VAD для лучшего разделения
                     "language": "ru" # Явно указываем язык
                 }

                response = requests.post(
                    "https://audio-prod.us-virginia-1.direct.fireworks.ai/v1/audio/transcriptions",
                    headers={"Authorization": f"Bearer {FIREWORKS_API_KEY}"},
                    files=files_data,
                    data=payload_data,
                    timeout=90 # Увеличим таймаут для потенциально долгой обработки v3
                )

            print(f"API Response Status Code: {response.status_code}")
            # print("API Response Headers:", response.headers)
            # print("API Response Body:", response.text) # Логгируем тело ответа для дебага

            response.raise_for_status() # Выбросит исключение для кодов 4xx/5xx

            result = response.json()
            # print("API Response JSON:", result) # Дебаг JSON
            transcribed_text = result.get('text', '').strip()

            if transcribed_text:
                # Передаем результат в основной поток через сигнал
                self.update_transcript.emit(transcribed_text)
                # Запускаем имитацию ввода через сигнал
                self.signal_simulate_typing.emit(transcribed_text + " ")
            else:
                # Сообщаем, если текст пустой (может быть нормально при тишине)
                print("Transcription result is empty.")
                self.signal_update_info.emit("Распознан пустой текст (тишина?)")
                # Сбрасываем сообщение через 2 секунды
                QTimer.singleShot(2000, lambda: self.update_info_label("Готов к записи") if self.program_running else None)


        except requests.exceptions.Timeout:
             error_message = "Ошибка API: Превышен таймаут ожидания ответа."
             print(error_message)
             self.signal_update_error.emit(error_message)
        except requests.exceptions.HTTPError as e:
            # Ошибка уже содержит код и тело ответа (если есть)
            error_message = f"Ошибка API ({e.response.status_code}): {e.response.text}"
            print(error_message)
            self.signal_update_error.emit(error_message)
        except requests.exceptions.RequestException as e:
             # Другие сетевые ошибки (DNS, соединение и т.д.)
             error_message = f"Ошибка сети при запросе к API: {str(e)}"
             print(error_message)
             self.signal_update_error.emit(error_message)
        except Exception as e:
            # Неожиданные ошибки (например, при парсинге JSON ответа)
            error_message = f"Неожиданная ошибка при обработке ответа API: {str(e)}"
            print(error_message)
            self.signal_update_error.emit(error_message)


    @pyqtSlot(str) # Слот для имитации ввода, вызывается из основного потока
    def simulate_typing_slot(self, text):
         # Запускаем саму симуляцию в отдельном потоке, чтобы не блокировать GUI
         threading.Thread(target=self._simulate_typing_worker, args=(text,), daemon=True).start()

    def _simulate_typing_worker(self, text):
        """Рабочая функция для симуляции ввода (выполняется в своем потоке)."""
        print(f"Simulating typing: '{text[:50]}...'")
        try:
            for char in text:
                self.kbd.type(char)
                time.sleep(0.015) # Немного увеличим задержку для стабильности
        except Exception as e:
            print(f"Error during typing simulation: {e}")


    @pyqtSlot(str) # Слот для обновления текста транскрипции
    def update_transcript_text(self, text):
        current_text = self.text_edit.toPlainText()
        # Добавляем новый текст с новой строки
        new_text = f"{current_text}\n{text}" if current_text else text
        self.text_edit.setText(new_text.strip())
        # Прокрутка к последней строке
        self.text_edit.moveCursor(QTextCursor.MoveOperation.End)
        self.info_label.setText("Транскрипция добавлена")
        # Сбрасываем сообщение через 2 секунды
        QTimer.singleShot(2000, lambda: self.update_info_label("Готов к записи") if self.program_running else None)

    @pyqtSlot(str) # Слот для отображения ошибок
    def update_transcript_error(self, error_message):
        # Используем HTML для красного цвета
        self.text_edit.append(f"<font color='#FF6B6B'><b>Ошибка:</b> {error_message}</font>")
        # Прокрутка к последней строке
        self.text_edit.moveCursor(QTextCursor.MoveOperation.End)
        self.info_label.setText("Произошла ошибка")
        # Держим сообщение об ошибке подольше
        QTimer.singleShot(5000, lambda: self.update_info_label("Готов к записи") if self.program_running else None)

    @pyqtSlot(str) # Слот для обновления info_label
    def update_info_label(self, text):
        if self.program_running: # Не обновлять, если закрываемся
             self.info_label.setText(text)

    def copy_text(self):
        clipboard = QApplication.clipboard()
        clipboard.setText(self.text_edit.toPlainText())
        self.info_label.setText("Текст скопирован")
        QTimer.singleShot(2000, lambda: self.update_info_label("Готов к записи") if self.program_running else None)

    def closeEvent(self, event):
        """Обработчик события закрытия окна."""
        print("Close event received. Shutting down...")
        self.program_running = False # Сигнал для остановки потоков

        # Останавливаем слушатель клавиатуры (он демон, но лучше явно)
        if hasattr(self, 'keyboard_listener') and self.keyboard_listener.is_alive():
             print("Stopping keyboard listener...")
             # Не используем keyboard_listener.stop() из другого потока,
             # он должен остановиться сам при выходе из on_press по return False
             # или при завершении программы, т.к. он демон.

        # Останавливаем аудиопоток
        self.stop_audio_stream()

        # Освобождаем ресурсы PyAudio
        if hasattr(self, 'p'):
             print("Terminating PyAudio...")
             self.p.terminate()

        # Очистка временной папки (опционально, система сама должна чистить temp)
        # Но если папка не в системном temp, лучше почистить
        if hasattr(self, 'temp_dir') and "whisper_gui_temp" in self.temp_dir or "temp_audio" in self.temp_dir:
             print(f"Cleaning up temp directory: {self.temp_dir}")
             try:
                 # Удаляем файлы внутри папки
                 for filename in os.listdir(self.temp_dir):
                     file_path = os.path.join(self.temp_dir, filename)
                     try:
                         if os.path.isfile(file_path) or os.path.islink(file_path):
                             os.unlink(file_path)
                     except Exception as e_del:
                         print(f'Failed to delete {file_path}. Reason: {e_del}')
                 # Пытаемся удалить саму папку, если она пуста
                 # os.rmdir(self.temp_dir) # Может не удалиться, если используется
             except Exception as e_clean:
                 print(f"Error during temp directory cleanup: {e_clean}")


        print("Shutdown complete. Accepting close event.")
        event.accept() # Принимаем событие закрытия

if __name__ == '__main__':
    app = QApplication(sys.argv)
    app.setStyle('Fusion') # Стиль Fusion часто выглядит лучше на разных ОС

    # Настройка темной палитры (как в предыдущем примере)
    dark_palette = QPalette()
    dark_palette.setColor(QPalette.ColorRole.Window, QColor(30, 30, 30))
    dark_palette.setColor(QPalette.ColorRole.WindowText, Qt.GlobalColor.white)
    dark_palette.setColor(QPalette.ColorRole.Base, QColor(42, 42, 42)) # Фон QTextEdit
    dark_palette.setColor(QPalette.ColorRole.AlternateBase, QColor(66, 66, 66))
    dark_palette.setColor(QPalette.ColorRole.ToolTipBase, Qt.GlobalColor.black) # Темный фон подсказок
    dark_palette.setColor(QPalette.ColorRole.ToolTipText, Qt.GlobalColor.white)
    dark_palette.setColor(QPalette.ColorRole.Text, Qt.GlobalColor.white) # Основной текст
    dark_palette.setColor(QPalette.ColorRole.Button, QColor(58, 58, 58)) # Фон кнопок
    dark_palette.setColor(QPalette.ColorRole.ButtonText, Qt.GlobalColor.white) # Текст на кнопках
    dark_palette.setColor(QPalette.ColorRole.BrightText, Qt.GlobalColor.red)
    dark_palette.setColor(QPalette.ColorRole.Link, QColor(42, 130, 218))
    dark_palette.setColor(QPalette.ColorRole.Highlight, QColor(65, 65, 65)) # Цвет выделения текста
    dark_palette.setColor(QPalette.ColorRole.HighlightedText, Qt.GlobalColor.white) # Цвет выделенного текста
    # Цвет для неактивных элементов
    dark_palette.setColor(QPalette.ColorGroup.Disabled, QPalette.ColorRole.ButtonText, QColor(127, 127, 127))
    dark_palette.setColor(QPalette.ColorGroup.Disabled, QPalette.ColorRole.WindowText, QColor(127, 127, 127))
    dark_palette.setColor(QPalette.ColorGroup.Disabled, QPalette.ColorRole.Text, QColor(127, 127, 127))

    app.setPalette(dark_palette)

    # Применяем стили CSS дополнительно для тонкой настройки
    app.setStyleSheet("""
        QWidget { /* Общие стили для всех виджетов */
            font-family: 'Segoe UI', Arial, sans-serif; /* Предпочтительный шрифт */
        }
        QTextEdit {
            background-color: #2A2A2A; /* Темнее фона окна */
            color: #E0E0E0;
            border: 1px solid #3A3A3A; /* Тонкая рамка */
            border-radius: 4px; /* Скругление углов */
            padding: 5px;
            selection-background-color: #0078D7; /* Синий цвет выделения */
            selection-color: white; /* Белый текст на выделении */
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
            background: #606060; /* Немного светлее при наведении */
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
            min-width: 70px; /* Минимальная ширина кнопки */
            font-size: 12px;
        }
        QPushButton:hover {
            background-color: #4A4A4A; /* Светлее при наведении */
            border: 1px solid #606060;
        }
        QPushButton:pressed {
            background-color: #2A2A2A; /* Темнее при нажатии */
            border: 1px solid #404040;
        }
        QPushButton:disabled { /* Стиль для неактивной кнопки */
             background-color: #2D2D2D;
             color: #6A6A6A;
             border: 1px solid #404040;
        }
        QLabel {
            /* Можно оставить цвет из палитры или задать явно */
             /* color: #FFFFFF; */
             font-size: 12px; /* Размер шрифта для Label */
        }
        #info_label QLabel, #shortcut_label QLabel { /* Применяем ID для специфичных стилей если нужно*/
            color: #888888;
            font-size: 11px;
        }
        QMainWindow {
             /* Можно оставить цвет фона из палитры */
             /* background-color: #1E1E1E; */
        }
    """)


    ex = WhisperGUI()
    ex.show()
    sys.exit(app.exec())