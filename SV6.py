import ctypes
from ctypes import wintypes
import os
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
from PyQt6.QtCore import QEasingCurve, QMetaObject, QPoint, QPropertyAnimation, QTimer, Qt, pyqtSignal, pyqtSlot
from PyQt6.QtGui import QColor, QFont, QLinearGradient, QPainter, QPen
from PyQt6.QtWidgets import QApplication, QComboBox, QFrame, QGraphicsDropShadowEffect, QHBoxLayout, QLabel, QMainWindow, QPushButton, QVBoxLayout, QWidget


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
LIVE_CHUNK_SECONDS = 1.8
LIVE_OVERLAP_SECONDS = 0.35


class Waveform(QWidget):
    """A small, animated input-level visualizer for the overlay."""

    def __init__(self):
        super().__init__()
        self.level = 0.0
        self.phase = 0.0
        self.recording = False
        self.setFixedHeight(34)
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.animate)
        self.timer.start(35)

    def set_recording(self, recording):
        self.recording = recording
        self.update()

    def set_level(self, level):
        self.level = max(0.0, min(1.0, level))

    def animate(self):
        self.phase += 0.22
        if not self.recording:
            self.level *= 0.86
        self.update()

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        width, height = self.width(), self.height()
        bars = 28
        bar_width = max(2, width // (bars * 2))
        gap = (width - bars * bar_width) / max(1, bars - 1)
        gradient = QLinearGradient(0, 0, width, 0)
        gradient.setColorAt(0, QColor("#8B5CF6"))
        gradient.setColorAt(1, QColor("#38BDF8"))
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(gradient)
        for index in range(bars):
            ripple = (np.sin(self.phase + index * 0.63) + 1.0) / 2.0
            amplitude = 3 + (self.level * (8 + 17 * ripple) if self.recording else 1)
            x = index * (bar_width + gap)
            painter.drawRoundedRect(int(x), int((height - amplitude) / 2), bar_width, int(amplitude), 2, 2)


class SallySpeechV6(QMainWindow):
    update_status = pyqtSignal(str)
    update_preview = pyqtSignal(str)
    update_wave = pyqtSignal(float)
    request_paste = pyqtSignal(str)

    def __init__(self):
        super().__init__()
        self.is_recording = False
        self.program_running = True
        self.pressed_keys = set()
        self.hotkey_active = False
        self.stream = None
        self.audio_buffer = []
        self.last_live_byte = 0
        self.live_history_words = []
        self.live_has_inserted = False
        self.active_engine = GIGAAM_ENGINE
        self.target_window = 0
        self.gigaam_model = None
        self.gigaam_load_lock = threading.Lock()
        self.gigaam_inference_lock = threading.Lock()
        self.groq_client = Groq(api_key=GROQ_API_KEY, timeout=90.0) if GROQ_API_KEY else None

        self.update_status.connect(self.status_label.setText)
        self.update_preview.connect(self.set_preview)
        self.update_wave.connect(self.waveform.set_level)
        self.request_paste.connect(self.paste_phrase)
        self.init_ui()
        self.setup_audio()

    def init_ui(self):
        self.setWindowTitle("Sally Speech 6")
        self.setFixedSize(430, 116)
        self.setWindowFlags(
            Qt.WindowType.FramelessWindowHint
            | Qt.WindowType.WindowStaysOnTopHint
            | Qt.WindowType.Tool
            | Qt.WindowType.WindowDoesNotAcceptFocus
        )
        self.setAttribute(Qt.WidgetAttribute.WA_ShowWithoutActivating)

        root = QWidget()
        self.setCentralWidget(root)
        root.setObjectName("root")
        root.setStyleSheet("""
            #root { background: #171923; border: 1px solid #363B52; border-radius: 18px; }
            QLabel { color: #E9ECF5; }
            QComboBox { background: #252A3A; color: #C9D1E8; border: 0; border-radius: 8px; padding: 4px 8px; font-size: 10px; }
            QComboBox::drop-down { border: 0; width: 14px; }
        """)
        shadow = QGraphicsDropShadowEffect(self)
        shadow.setBlurRadius(28)
        shadow.setOffset(0, 8)
        shadow.setColor(QColor(0, 0, 0, 150))
        root.setGraphicsEffect(shadow)

        layout = QVBoxLayout(root)
        layout.setContentsMargins(16, 12, 16, 12)
        layout.setSpacing(3)
        top = QHBoxLayout()
        top.setSpacing(9)

        self.record_button = QPushButton("●")
        self.record_button.setAccessibleName("Начать или остановить запись")
        self.record_button.setFixedSize(38, 38)
        self.record_button.setCursor(Qt.CursorShape.PointingHandCursor)
        self.record_button.clicked.connect(self.toggle_recording)
        self.record_button.setStyleSheet(self.button_style(False))
        top.addWidget(self.record_button)

        labels = QVBoxLayout()
        self.status_label = QLabel("Готово · Правый Ctrl или кнопка")
        self.status_label.setFont(QFont("Segoe UI", 10, QFont.Weight.DemiBold))
        labels.addWidget(self.status_label)
        self.preview_label = QLabel("Локальный режим вставляет фразы во время речи")
        self.preview_label.setStyleSheet("color: #9AA4BD; font-size: 10px;")
        self.preview_label.setMaximumWidth(245)
        labels.addWidget(self.preview_label)
        top.addLayout(labels, 1)

        self.engine_combo = QComboBox()
        self.engine_combo.addItem("GigaAM GPU", GIGAAM_ENGINE)
        self.engine_combo.addItem("Groq", GROQ_ENGINE)
        self.engine_combo.currentIndexChanged.connect(self.engine_changed)
        top.addWidget(self.engine_combo)
        layout.addLayout(top)

        self.waveform = Waveform()
        layout.addWidget(self.waveform)

    def button_style(self, recording):
        color = "#EF4444" if recording else "#7C3AED"
        hover = "#F87171" if recording else "#8B5CF6"
        return f"""QPushButton {{ background: {color}; color: white; border: 0; border-radius: 19px; font-size: 20px; padding-bottom: 2px; }}
                   QPushButton:hover {{ background: {hover}; }}"""

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
                self.update_status.emit("GigaAM GPU · вставка во время речи")
            except RuntimeError as e:
                self.engine_combo.blockSignals(True)
                self.engine_combo.setCurrentIndex(1)
                self.engine_combo.blockSignals(False)
                self.update_status.emit(str(e))
        else:
            self.update_status.emit("Groq · вставка после окончания записи")

    def require_cuda(self):
        if torch is None:
            raise RuntimeError(f"CUDA PyTorch недоступен: {TORCH_IMPORT_ERROR}")
        if not torch.cuda.is_available():
            raise RuntimeError("GigaAM отключён: CUDA недоступна. Режим CPU не используется.")
        return torch.cuda.get_device_name(torch.cuda.current_device())

    def capture_target_window(self):
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
        self.capture_target_window()
        try:
            self.stream = self.p.open(format=pyaudio.paInt16, channels=CHANNELS, rate=RATE, input=True, frames_per_buffer=CHUNK)
        except Exception as e:
            self.update_status.emit(f"Микрофон недоступен: {e}")
            return
        self.audio_buffer = []
        self.active_engine = self.selected_engine()
        self.last_live_byte = 0
        self.live_history_words = []
        self.live_has_inserted = False
        self.next_live_at = time.monotonic() + LIVE_CHUNK_SECONDS
        self.is_recording = True
        self.waveform.set_recording(True)
        self.record_button.setText("■")
        self.record_button.setStyleSheet(self.button_style(True))
        self.update_status.emit("Слушаю…")
        self.update_preview.emit("Говорите — фразы появятся в активном окне")

    def stop_recording(self):
        self.is_recording = False
        self.waveform.set_recording(False)
        self.record_button.setText("●")
        self.record_button.setStyleSheet(self.button_style(False))
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
        threading.Thread(target=self.transcribe_final, args=(audio, self.active_engine, self.live_has_inserted), daemon=True).start()

    def record_loop(self):
        while self.program_running:
            if not self.is_recording or not self.stream:
                time.sleep(0.02)
                continue
            try:
                data = self.stream.read(CHUNK, exception_on_overflow=False)
                self.audio_buffer.append(data)
                samples = np.frombuffer(data, dtype=np.int16).astype(np.float32)
                self.update_wave.emit(float(min(1.0, np.sqrt(np.mean(samples * samples)) / 8500.0)))
                if self.active_engine == GIGAAM_ENGINE and time.monotonic() >= self.next_live_at:
                    self.queue_live_chunk()
                    self.next_live_at += LIVE_CHUNK_SECONDS
            except Exception as e:
                self.update_status.emit(f"Ошибка записи: {e}")
                self.is_recording = False

    def queue_live_chunk(self):
        all_audio = b"".join(self.audio_buffer)
        bytes_per_second = RATE * CHANNELS * 2
        start = max(0, self.last_live_byte - int(LIVE_OVERLAP_SECONDS * bytes_per_second))
        segment = all_audio[start:]
        self.last_live_byte = len(all_audio)
        if len(segment) >= int(0.5 * bytes_per_second):
            threading.Thread(target=self.transcribe_live_chunk, args=(segment,), daemon=True).start()

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

    def transcribe_live_chunk(self, audio_data):
        wav_file = None
        try:
            wav_file = self.write_wav(audio_data)
            with self.gigaam_inference_lock:
                text = (self.get_gigaam_model().transcribe(wav_file).text or "").strip()
            new_text = self.only_new_words(text)
            if new_text:
                self.live_has_inserted = True
                self.update_preview.emit(new_text)
                self.request_paste.emit(new_text + " ")
        except Exception as e:
            self.update_status.emit(f"GigaAM: {e}")
        finally:
            if wav_file and os.path.exists(wav_file):
                os.remove(wav_file)

    def transcribe_gigaam_file(self, model, wav_file):
        """The upstream short-form GigaAM API accepts at most 25 seconds per call."""
        with wave.open(wav_file, "rb") as source:
            max_frames = source.getframerate() * 24
            if source.getnframes() <= max_frames:
                return (model.transcribe(wav_file).text or "").strip()
            params = source.getparams()
            texts = []
            while True:
                frames = source.readframes(max_frames)
                if not frames:
                    break
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

    def only_new_words(self, text):
        words = text.split()
        if not words:
            return ""
        history = self.live_history_words
        overlap = 0
        for size in range(min(len(history), len(words)), 0, -1):
            if [w.lower() for w in history[-size:]] == [w.lower() for w in words[:size]]:
                overlap = size
                break
        new_words = words[overlap:]
        self.live_history_words.extend(new_words)
        return " ".join(new_words)

    def transcribe_final(self, audio_data, engine, live_inserted):
        wav_file = None
        try:
            wav_file = self.write_wav(audio_data)
            if engine == GIGAAM_ENGINE:
                with self.gigaam_inference_lock:
                    text = self.transcribe_gigaam_file(self.get_gigaam_model(), wav_file)
                self.update_preview.emit(text or "Тишина")
                if text and not live_inserted:
                    self.request_paste.emit(text + " ")
            else:
                text = self.transcribe_groq(wav_file)
                if text:
                    self.update_preview.emit(text)
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
        self.preview_label.setText(text[-95:])

    @pyqtSlot(str)
    def paste_phrase(self, text):
        clipboard = QApplication.clipboard()
        previous = clipboard.text()
        clipboard.setText(text)
        if os.name == "nt" and self.target_window:
            user32 = ctypes.WinDLL("user32", use_last_error=True)
            user32.SetForegroundWindow(self.target_window)
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
