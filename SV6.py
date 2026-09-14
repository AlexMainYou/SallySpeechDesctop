import ctypes
from ctypes import wintypes
import os
import shutil
import subprocess
import sys
import threading
import time
import uuid
import wave

try:
    # PyQt6 can load DLLs that conflict with CUDA PyTorch on Windows.
    import torch
    TORCH_IMPORT_ERROR = None
except (ImportError, OSError) as e:
    torch = None
    TORCH_IMPORT_ERROR = e

import numpy as np
import pyaudio
from groq import APIConnectionError, APIStatusError, APITimeoutError, Groq
from pynput import keyboard
from PyQt6.QtCore import QMetaObject, QTimer, Qt, pyqtSignal, pyqtSlot
from PyQt6.QtGui import QColor, QFont, QLinearGradient, QPainter, QPen
from PyQt6.QtWidgets import QApplication, QComboBox, QFileDialog, QGraphicsDropShadowEffect, QHBoxLayout, QLabel, QMainWindow, QPushButton, QTextEdit, QVBoxLayout, QWidget


def get_app_dir():
    return os.path.dirname(sys.executable) if getattr(sys, "frozen", False) else os.path.dirname(os.path.abspath(__file__))


def get_gigaam_cache_dir():
    cache_base = os.environ.get("LOCALAPPDATA") or get_app_dir()
    return os.path.join(cache_base, "SallySpeech", "model_cache", "gigaam")


def load_env_value(name, default=None):
    try:
        with open(os.path.join(get_app_dir(), ".env"), "r", encoding="utf-8") as env_file:
            for raw_line in env_file:
                line = raw_line.strip()
                if line and not line.startswith("#") and "=" in line:
                    key, value = line.split("=", 1)
                    if key.strip() == name:
                        return value.strip().strip('"').strip("'")
    except OSError:
        pass
    return os.environ.get(name, default)


GROQ_API_KEY = load_env_value("GROQ_API_KEY")
GROQ_ENGINE = "groq"
GIGAAM_ENGINE = "gigaam"
GROQ_MODEL = "whisper-large-v3"
GIGAAM_MODEL = "v3_e2e_rnnt"
RATE = 44100
CHANNELS = 1
CHUNK = 1024
class RecordButton(QPushButton):
    """Round microphone button drawn as a real icon instead of a text glyph."""

    def __init__(self):
        super().__init__()
        self.recording = False
        self.setFixedSize(46, 46)
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.setToolTip("Начать или остановить запись")
        self.setStyleSheet("QPushButton { border: 0; background: transparent; }")
        self.phase = 0.0
        self.animation_timer = QTimer(self)
        self.animation_timer.timeout.connect(self.animate)
        self.animation_timer.start(6)

    def set_recording(self, recording):
        self.recording = recording
        self.update()

    def animate(self):
        if self.recording:
            self.phase += 0.045
            self.update()

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        color = QColor("#F0526D") if self.recording else QColor("#7C5CFC")
        diameter = 38
        if self.recording:
            diameter += int((np.sin(self.phase) + 1) * 2)
        offset = (46 - diameter) // 2
        painter.setBrush(color)
        painter.setPen(Qt.PenStyle.NoPen)
        painter.drawEllipse(offset, offset, diameter, diameter)
        painter.setPen(QPen(QColor("white"), 2.4, Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap))
        if self.recording:
            painter.setBrush(QColor("white"))
            painter.drawRoundedRect(17, 17, 12, 12, 2, 2)
        else:
            painter.setBrush(QColor("white"))
            painter.drawRoundedRect(17, 9, 12, 19, 6, 6)
            painter.drawLine(23, 34, 23, 30)
            painter.drawLine(18, 36, 28, 36)


class MediaButton(QPushButton):
    """Small document button that does not depend on a font glyph."""

    def __init__(self):
        super().__init__()
        self.setFixedSize(28, 28)
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.setToolTip("Открыть расшифровку файла")
        self.setStyleSheet("QPushButton { border: 0; background: #252A3A; border-radius: 8px; } QPushButton:hover { background: #38405A; }")

    def paintEvent(self, event):
        super().paintEvent(event)
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.setPen(QPen(QColor("#DDE4F7"), 1.5, Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap))
        painter.setBrush(Qt.BrushStyle.NoBrush)
        painter.drawRoundedRect(8, 5, 12, 17, 2, 2)
        painter.drawLine(16, 5, 20, 9)
        painter.drawLine(16, 5, 16, 9)
        painter.drawLine(16, 9, 20, 9)
        painter.drawLine(11, 13, 17, 13)
        painter.drawLine(11, 17, 17, 17)


class DropArea(QLabel):
    file_dropped = pyqtSignal(str)
    choose_requested = pyqtSignal()

    def __init__(self):
        super().__init__("Перетащите сюда\nаудио или видео\n\nили нажмите для выбора")
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setAcceptDrops(True)
        self.setMinimumWidth(180)
        self.setStyleSheet("""
            QLabel { color: #9CA9C7; background: #202536; border: 1px dashed #5C6684; border-radius: 18px; font-size: 12px; }
            QLabel:hover { background: #262D42; border-color: #8B5CF6; color: #DDE4F7; }
        """)

    def dragEnterEvent(self, event):
        if event.mimeData().hasUrls() and len(event.mimeData().urls()) == 1 and event.mimeData().urls()[0].isLocalFile():
            event.acceptProposedAction()

    def dropEvent(self, event):
        self.file_dropped.emit(event.mimeData().urls()[0].toLocalFile())
        event.acceptProposedAction()

    def mousePressEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            self.choose_requested.emit()
        super().mousePressEvent(event)


class Waveform(QWidget):
    """A small, animated input-level visualizer for the overlay."""

    def __init__(self):
        super().__init__()
        self.target_level = 0.0
        self.display_level = 0.0
        self.phase = 0.0
        self.recording = False
        self.setFixedHeight(34)
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.animate)
        self.timer.start(6)

    def set_recording(self, recording):
        self.recording = recording
        self.update()

    def set_level(self, level):
        self.target_level = max(0.0, min(1.0, level))

    def animate(self):
        self.phase += 0.055
        self.display_level += (self.target_level - self.display_level) * 0.13
        if not self.recording:
            self.target_level = 0.0
            self.display_level *= 0.92
        self.update()

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        width, height = self.width(), self.height()
        bars = 34
        bar_width = max(2, width // (bars * 2 + 5))
        gap = (width - bars * bar_width) / max(1, bars - 1)
        gradient = QLinearGradient(0, 0, width, 0)
        gradient.setColorAt(0, QColor("#8B5CF6"))
        gradient.setColorAt(1, QColor("#38BDF8"))
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(gradient)
        for index in range(bars):
            ripple = (np.sin(self.phase + index * 0.63) + 1.0) / 2.0
            idle = 2 + ripple * 2
            amplitude = idle + (self.display_level * (10 + 18 * ripple) if self.recording else 0)
            x = index * (bar_width + gap)
            painter.drawRoundedRect(int(x), int((height - amplitude) / 2), bar_width, int(amplitude), 2, 2)


class SallySpeechV6(QMainWindow):
    update_status = pyqtSignal(str)
    update_preview = pyqtSignal(str)
    update_wave = pyqtSignal(float)
    request_paste = pyqtSignal(str)
    update_transcript = pyqtSignal(str)

    def __init__(self):
        super().__init__()
        self.is_recording = False
        self.program_running = True
        self.pressed_keys = set()
        self.hotkey_active = False
        self.stream = None
        self.audio_buffer = []
        self.active_engine = GIGAAM_ENGINE
        self.target_window = 0
        self.gigaam_model = None
        self.gigaam_load_lock = threading.Lock()
        self.gigaam_inference_lock = threading.Lock()
        self.media_running = False
        self.media_expanded = False
        self.groq_client = Groq(api_key=GROQ_API_KEY, timeout=90.0) if GROQ_API_KEY else None

        self.init_ui()
        self.update_status.connect(self.status_label.setText)
        self.update_preview.connect(self.set_preview)
        self.update_wave.connect(self.waveform.set_level)
        self.request_paste.connect(self.paste_phrase)
        self.update_transcript.connect(self.set_transcript)
        self.target_timer = QTimer(self)
        self.target_timer.timeout.connect(self.remember_target_window)
        self.target_timer.start(150)
        self.setup_audio()

    def init_ui(self):
        self.setWindowTitle("Sally Speech 6")
        # The transparent margin gives the drop shadow room to fade naturally.
        self.setFixedSize(478, 154)
        self.setWindowFlags(
            Qt.WindowType.FramelessWindowHint
            | Qt.WindowType.WindowStaysOnTopHint
            | Qt.WindowType.Tool
            | Qt.WindowType.WindowDoesNotAcceptFocus
        )
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground, True)
        self.setAttribute(Qt.WidgetAttribute.WA_ShowWithoutActivating)

        canvas = QWidget()
        canvas.setStyleSheet("background: transparent;")
        self.setCentralWidget(canvas)
        canvas_layout = QVBoxLayout(canvas)
        canvas_layout.setContentsMargins(16, 12, 16, 22)

        root = QWidget(canvas)
        root.setObjectName("root")
        root.setStyleSheet("""
            #root { background: #171923; border: 1px solid #363B52; border-radius: 33px; }
            QLabel { color: #E9ECF5; }
            QComboBox { background: #252A3A; color: #C9D1E8; border: 0; border-radius: 8px; padding: 4px 8px; font-size: 10px; }
            QComboBox::drop-down { border: 0; width: 14px; }
        """)
        shadow = QGraphicsDropShadowEffect(self)
        shadow.setBlurRadius(28)
        shadow.setOffset(0, 8)
        shadow.setColor(QColor(0, 0, 0, 150))
        root.setGraphicsEffect(shadow)
        canvas_layout.addWidget(root)

        layout = QVBoxLayout(root)
        layout.setContentsMargins(16, 12, 16, 10)
        layout.setSpacing(3)
        top = QHBoxLayout()
        top.setSpacing(9)

        self.record_button = RecordButton()
        self.record_button.setAccessibleName("Начать или остановить запись")
        self.record_button.clicked.connect(self.toggle_recording)
        top.addWidget(self.record_button)

        labels = QVBoxLayout()
        self.status_label = QLabel("Готово · Правый Ctrl или кнопка")
        self.status_label.setFont(QFont("Segoe UI", 10, QFont.Weight.DemiBold))
        labels.addWidget(self.status_label)
        self.preview_label = QLabel("Локальный режим вставляет фразы во время речи")
        self.preview_label.setStyleSheet("color: #9AA4BD; font-size: 10px;")
        self.preview_label.setMaximumWidth(220)
        labels.addWidget(self.preview_label)
        top.addLayout(labels, 1)

        self.engine_combo = QComboBox()
        self.engine_combo.addItem("GigaAM GPU", GIGAAM_ENGINE)
        self.engine_combo.addItem("Groq", GROQ_ENGINE)
        self.engine_combo.currentIndexChanged.connect(self.engine_changed)
        top.addWidget(self.engine_combo)

        self.media_button = MediaButton()
        self.media_button.clicked.connect(self.toggle_media_panel)
        top.addWidget(self.media_button)

        self.close_button = QPushButton("×")
        self.close_button.setAccessibleName("Закрыть Sally Speech")
        self.close_button.setToolTip("Закрыть программу")
        self.close_button.setFixedSize(28, 28)
        self.close_button.setCursor(Qt.CursorShape.PointingHandCursor)
        self.close_button.setStyleSheet("""
            QPushButton { color: #AEB7D0; background: #252A3A; border: 0; border-radius: 8px; font-size: 20px; font-weight: 300; padding-bottom: 3px; }
            QPushButton:hover { color: white; background: #D74B63; }
        """)
        self.close_button.clicked.connect(self.close)
        top.addWidget(self.close_button)
        layout.addLayout(top)

        self.waveform = Waveform()
        layout.addWidget(self.waveform)

        self.media_panel = QWidget()
        media_layout = QHBoxLayout(self.media_panel)
        media_layout.setContentsMargins(0, 10, 0, 0)
        media_layout.setSpacing(10)
        self.drop_area = DropArea()
        self.drop_area.file_dropped.connect(self.start_media_transcription)
        self.drop_area.choose_requested.connect(self.choose_media_file)
        media_layout.addWidget(self.drop_area, 4)

        transcript_column = QVBoxLayout()
        self.transcript_edit = QTextEdit()
        self.transcript_edit.setReadOnly(True)
        self.transcript_edit.setPlaceholderText("Здесь появится последняя расшифровка")
        self.transcript_edit.setStyleSheet("""
            QTextEdit { color: #E9ECF5; background: #202536; border: 1px solid #363B52; border-radius: 18px; padding: 8px; font-size: 11px; }
            QScrollBar:vertical { background: transparent; width: 10px; margin: 10px 3px 10px 0; }
            QScrollBar::handle:vertical { background: #697493; min-height: 34px; border-radius: 5px; }
            QScrollBar::handle:vertical:hover { background: #8B98BC; }
            QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical { height: 0px; }
            QScrollBar::add-page:vertical, QScrollBar::sub-page:vertical { background: transparent; }
        """)
        transcript_column.addWidget(self.transcript_edit, 1)
        transcript_actions = QHBoxLayout()
        self.copy_button = QPushButton("Копировать")
        self.copy_button.clicked.connect(self.copy_transcript)
        self.save_button = QPushButton("Сохранить…")
        self.save_button.clicked.connect(self.save_transcript)
        for button in (self.copy_button, self.save_button):
            button.setCursor(Qt.CursorShape.PointingHandCursor)
            button.setStyleSheet("QPushButton { color: #DDE4F7; background: #2A3147; border: 0; border-radius: 9px; padding: 6px 10px; font-size: 11px; } QPushButton:hover { background: #3A4666; }")
            transcript_actions.addWidget(button, 1)
        transcript_column.addLayout(transcript_actions)
        media_layout.addLayout(transcript_column, 5)
        self.media_panel.setVisible(False)
        layout.addWidget(self.media_panel, 1)

    def setup_audio(self):
        try:
            self.p = pyaudio.PyAudio()
        except Exception as e:
            raise RuntimeError(f"Не удалось инициализировать микрофон: {e}")
        self.keyboard_listener = keyboard.Listener(on_press=self.on_press, on_release=self.on_release, daemon=True)
        self.keyboard_listener.start()
        self.record_thread = threading.Thread(target=self.record_loop, daemon=True)
        self.record_thread.start()

    def on_press(self, key):
        self.pressed_keys.add(key)
        if keyboard.Key.ctrl_r in self.pressed_keys and not self.hotkey_active:
            self.hotkey_active = True
            QMetaObject.invokeMethod(self, "toggle_recording", Qt.ConnectionType.QueuedConnection)

    def on_release(self, key):
        self.pressed_keys.discard(key)
        if keyboard.Key.ctrl_r not in self.pressed_keys:
            self.hotkey_active = False

    def selected_engine(self):
        return self.engine_combo.currentData()

    def engine_changed(self, _index=None):
        if self.selected_engine() == GIGAAM_ENGINE:
            try:
                self.require_cuda()
                self.update_status.emit("GigaAM GPU · вставка после записи")
            except RuntimeError as e:
                self.engine_combo.blockSignals(True)
                self.engine_combo.setCurrentIndex(1)
                self.engine_combo.blockSignals(False)
                self.update_status.emit(str(e))
        else:
            self.update_status.emit("Groq · вставка после окончания записи")

    @pyqtSlot()
    def toggle_media_panel(self):
        self.media_expanded = not self.media_expanded
        self.media_panel.setVisible(self.media_expanded)
        self.setFixedSize(478, 478 if self.media_expanded else 154)
        self.media_button.setToolTip("Скрыть расшифровку файла" if self.media_expanded else "Открыть расшифровку файла")

    @pyqtSlot()
    def choose_media_file(self):
        source, _ = QFileDialog.getOpenFileName(
            self,
            "Выберите аудио или видео",
            "",
            "Медиафайлы (*.mp3 *.wav *.m4a *.aac *.ogg *.flac *.opus *.mp4 *.mkv *.avi *.mov *.webm *.3gp);;Все файлы (*)",
        )
        if source:
            self.start_media_transcription(source)

    @pyqtSlot(str)
    def start_media_transcription(self, source):
        if self.media_running:
            self.update_status.emit("Предыдущий файл ещё распознаётся")
            return
        if not os.path.isfile(source):
            self.update_status.emit("Файл не найден")
            return
        self.media_running = True
        self.drop_area.setText(f"Обрабатываю\n{os.path.basename(source)}")
        self.update_transcript.emit("")
        threading.Thread(target=self.transcribe_media, args=(source,), daemon=True).start()

    def find_ffmpeg(self):
        bundled = os.path.join(get_app_dir(), "ffmpeg.exe")
        return bundled if os.path.isfile(bundled) else shutil.which("ffmpeg")

    def convert_media_to_wav(self, source):
        ffmpeg = self.find_ffmpeg()
        if not ffmpeg:
            raise RuntimeError("FFmpeg не найден. Добавьте ffmpeg.exe в PATH или рядом с программой.")
        output = os.path.join(get_app_dir(), "temp_audio", f"sv6_media_{uuid.uuid4()}.wav")
        os.makedirs(os.path.dirname(output), exist_ok=True)
        result = subprocess.run(
            [ffmpeg, "-y", "-i", source, "-vn", "-ac", "1", "-ar", str(RATE), "-c:a", "pcm_s16le", output],
            capture_output=True,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        )
        if result.returncode != 0 or not os.path.exists(output):
            details = result.stderr.decode(errors="replace").strip().splitlines()
            raise RuntimeError(details[-1] if details else "FFmpeg не смог прочитать этот файл")
        return output

    def transcribe_media(self, source):
        wav_file = None
        try:
            self.update_status.emit("Подготавливаю аудио через FFmpeg…")
            wav_file = self.convert_media_to_wav(source)
            self.update_status.emit("Загружаю GigaAM GPU…")
            with self.gigaam_inference_lock:
                text = self.transcribe_gigaam_file(self.get_gigaam_model(), wav_file, "Распознаю файл")
            self.update_transcript.emit(text or "В файле не удалось найти речь.")
            self.update_status.emit("Файл распознан · текст можно скопировать или сохранить")
        except Exception as e:
            self.update_transcript.emit(f"Ошибка: {e}")
            self.update_status.emit("Не удалось распознать файл")
        finally:
            if wav_file and os.path.exists(wav_file):
                os.remove(wav_file)
            self.media_running = False
            self.drop_area.setText("Перетащите сюда\nаудио или видео\n\nили нажмите для выбора")

    @pyqtSlot(str)
    def set_transcript(self, text):
        self.transcript_edit.setPlainText(text)

    @pyqtSlot()
    def copy_transcript(self):
        text = self.transcript_edit.toPlainText()
        if text:
            QApplication.clipboard().setText(text)
            self.update_status.emit("Текст скопирован")

    @pyqtSlot()
    def save_transcript(self):
        text = self.transcript_edit.toPlainText()
        if not text:
            return
        filename, _ = QFileDialog.getSaveFileName(self, "Сохранить расшифровку", "transcript.txt", "Текстовый файл (*.txt)")
        if filename:
            if not filename.lower().endswith(".txt"):
                filename += ".txt"
            with open(filename, "w", encoding="utf-8") as output:
                output.write(text)
            self.update_status.emit("Текст сохранён")

    def require_cuda(self):
        if torch is None:
            raise RuntimeError(f"CUDA PyTorch недоступен: {TORCH_IMPORT_ERROR}")
        if not torch.cuda.is_available():
            raise RuntimeError("GigaAM отключён: CUDA недоступна. Режим CPU не используется.")
        return torch.cuda.get_device_name(torch.cuda.current_device())

    def remember_target_window(self):
        if os.name != "nt":
            return
        foreground = ctypes.WinDLL("user32", use_last_error=True).GetForegroundWindow()
        if foreground and foreground != int(self.winId()):
            self.target_window = foreground

    @pyqtSlot()
    def toggle_recording(self):
        if self.is_recording:
            self.stop_recording()
        else:
            self.start_recording()

    def start_recording(self):
        self.remember_target_window()
        try:
            self.stream = self.p.open(format=pyaudio.paInt16, channels=CHANNELS, rate=RATE, input=True, frames_per_buffer=CHUNK)
        except Exception as e:
            self.update_status.emit(f"Микрофон недоступен: {e}")
            return
        self.audio_buffer = []
        self.active_engine = self.selected_engine()
        self.is_recording = True
        self.waveform.set_recording(True)
        self.record_button.set_recording(True)
        self.update_status.emit("Слушаю…")
        self.update_preview.emit("Текст будет после остановки")
        self.update_transcript.emit("")

    def stop_recording(self):
        self.is_recording = False
        self.waveform.set_recording(False)
        self.record_button.set_recording(False)
        if self.stream:
            try:
                self.stream.stop_stream()
                self.stream.close()
            finally:
                self.stream = None
        audio = b"".join(self.audio_buffer)
        if not audio:
            self.update_status.emit("Запись пуста")
            return
        self.update_status.emit("Завершаю распознавание…")
        threading.Thread(target=self.transcribe_final, args=(audio, self.active_engine), daemon=True).start()

    def record_loop(self):
        while self.program_running:
            if not self.is_recording or not self.stream:
                time.sleep(0.02)
                continue
            try:
                data = self.stream.read(CHUNK, exception_on_overflow=False)
                self.audio_buffer.append(data)
                samples = np.frombuffer(data, dtype=np.int16).astype(np.float32)
                rms = np.sqrt(np.mean(samples * samples))
                level = min(1.0, max(0.04, (rms / 2600.0) ** 0.55))
                self.update_wave.emit(float(level))
            except Exception as e:
                self.update_status.emit(f"Ошибка записи: {e}")
                self.is_recording = False

    def write_wav(self, audio_data):
        os.makedirs(os.path.join(get_app_dir(), "temp_audio"), exist_ok=True)
        filename = os.path.join(get_app_dir(), "temp_audio", f"sv6_{uuid.uuid4()}.wav")
        with wave.open(filename, "wb") as wf:
            wf.setnchannels(CHANNELS)
            wf.setsampwidth(self.p.get_sample_size(pyaudio.paInt16))
            wf.setframerate(RATE)
            wf.writeframes(audio_data)
        return filename

    def get_gigaam_model(self):
        with self.gigaam_load_lock:
            if self.gigaam_model is None:
                device = self.require_cuda()
                self.update_status.emit(f"Загружаю GigaAM на {device}…")
                import gigaam
                self.gigaam_model = gigaam.load_model(GIGAAM_MODEL, device="cuda", fp16_encoder=True, download_root=get_gigaam_cache_dir())
                if next(self.gigaam_model.parameters()).device.type != "cuda":
                    self.gigaam_model = None
                    raise RuntimeError("GigaAM не загрузился на GPU")
            return self.gigaam_model

    def transcribe_gigaam_file(self, model, wav_file, task_name="Распознаю запись"):
        """The upstream short-form GigaAM API accepts at most 25 seconds per call."""
        with wave.open(wav_file, "rb") as source:
            max_frames = source.getframerate() * 24
            if source.getnframes() <= max_frames:
                self.update_status.emit(task_name + "…")
                return (model.transcribe(wav_file).text or "").strip()
            params = source.getparams()
            total_parts = (source.getnframes() + max_frames - 1) // max_frames
            texts = []
            part_number = 0
            while True:
                frames = source.readframes(max_frames)
                if not frames:
                    break
                part_number += 1
                self.update_status.emit(f"{task_name}: фрагмент {part_number}/{total_parts}")
                part = os.path.join(os.path.dirname(wav_file), f"sv6_part_{uuid.uuid4()}.wav")
                try:
                    with wave.open(part, "wb") as target:
                        target.setparams(params)
                        target.writeframes(frames)
                    text = (model.transcribe(part).text or "").strip()
                    if text:
                        texts.append(text)
                finally:
                    if os.path.exists(part):
                        os.remove(part)
            return " ".join(texts)

    def transcribe_final(self, audio_data, engine):
        wav_file = None
        try:
            wav_file = self.write_wav(audio_data)
            if engine == GIGAAM_ENGINE:
                with self.gigaam_inference_lock:
                    text = self.transcribe_gigaam_file(self.get_gigaam_model(), wav_file)
                self.update_preview.emit(text or "Тишина")
                self.update_transcript.emit(text or "")
                if text:
                    self.request_paste.emit(text + " ")
            else:
                text = self.transcribe_groq(wav_file)
                if text:
                    self.update_preview.emit(text)
                    self.update_transcript.emit(text)
                    self.request_paste.emit(text + " ")
            self.update_status.emit("Готово · Правый Ctrl или кнопка")
        except Exception as e:
            self.update_status.emit(f"Ошибка распознавания: {e}")
        finally:
            if wav_file and os.path.exists(wav_file):
                os.remove(wav_file)

    def transcribe_groq(self, wav_file):
        if not self.groq_client:
            raise RuntimeError("GROQ_API_KEY не задан")
        with open(wav_file, "rb") as audio_file:
            response = self.groq_client.audio.transcriptions.create(file=(os.path.basename(wav_file), audio_file.read()), model=GROQ_MODEL, temperature=0, language="ru")
        return (getattr(response, "text", "") or "").strip()

    @pyqtSlot(str)
    def set_preview(self, text):
        words = text.split()
        self.preview_label.setText(" ".join(words[-4:]))

    @pyqtSlot(str)
    def paste_phrase(self, text):
        clipboard = QApplication.clipboard()
        previous = clipboard.text()
        clipboard.setText(text)
        if os.name == "nt" and self.target_window:
            user32 = ctypes.WinDLL("user32", use_last_error=True)
            user32.SetForegroundWindow(self.target_window)
            time.sleep(0.05)
        self.send_ctrl_v()
        QTimer.singleShot(300, lambda: self.restore_clipboard(previous, text))

    def restore_clipboard(self, previous, pasted):
        clipboard = QApplication.clipboard()
        if clipboard.text() == pasted:
            clipboard.setText(previous)

    def send_ctrl_v(self):
        if os.name != "nt":
            return
        user32 = ctypes.WinDLL("user32", use_last_error=True)
        class KEYBDINPUT(ctypes.Structure):
            _fields_ = [("wVk", wintypes.WORD), ("wScan", wintypes.WORD), ("dwFlags", wintypes.DWORD), ("time", wintypes.DWORD), ("dwExtraInfo", ctypes.c_size_t)]
        class MOUSEINPUT(ctypes.Structure):
            _fields_ = [("dx", wintypes.LONG), ("dy", wintypes.LONG), ("mouseData", wintypes.DWORD), ("dwFlags", wintypes.DWORD), ("time", wintypes.DWORD), ("dwExtraInfo", ctypes.c_size_t)]
        class INPUT_UNION(ctypes.Union):
            _fields_ = [("ki", KEYBDINPUT), ("mi", MOUSEINPUT)]
        class INPUT(ctypes.Structure):
            _anonymous_ = ("u",)
            _fields_ = [("type", wintypes.DWORD), ("u", INPUT_UNION)]
        def key(vk, up=False):
            return INPUT(type=1, ki=KEYBDINPUT(vk, 0, 0x0002 if up else 0, 0, 0))
        events = (INPUT * 4)(key(0x11), key(0x56), key(0x56, True), key(0x11, True))
        if user32.SendInput(4, events, ctypes.sizeof(INPUT)) != 4:
            raise ctypes.WinError(ctypes.get_last_error())

    def mousePressEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            self.drag_position = event.globalPosition().toPoint() - self.frameGeometry().topLeft()
            event.accept()

    def mouseMoveEvent(self, event):
        if event.buttons() & Qt.MouseButton.LeftButton and hasattr(self, "drag_position"):
            self.move(event.globalPosition().toPoint() - self.drag_position)
            event.accept()

    def closeEvent(self, event):
        self.program_running = False
        if self.stream:
            self.stream.close()
        self.p.terminate()
        event.accept()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    window = SallySpeechV6()
    window.show()
    sys.exit(app.exec())
