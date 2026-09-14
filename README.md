# SallySpeechDesctop

SallySpeechDesctop is a small Windows desktop dictation tool. Press a global hotkey, speak, press it again, and the recognized Russian text is inserted into the active application.

The app supports Groq Whisper (`whisper-large-v3`) and local GigaAM v3 recognition on an NVIDIA GPU.

## Features

- Global recording hotkey: right `Ctrl`
- Exit hotkey: `End`
- Groq Whisper transcription
- Local GigaAM v3 e2e RNNT transcription on CUDA; the model is downloaded automatically on first use
- Engine selector: Groq Whisper or GigaAM v3 (local GPU)
- Direct Unicode text insertion without overwriting the clipboard
- Local transcript window with manual copy button
- Short-lived audio files stored in `temp_audio/`
- Packaged Windows executable available in GitHub Releases

## Quick Start

1. Download `SV5.exe` from the latest release.
2. Put a `.env` file next to `SV5.exe`.
3. Add your Groq API key:

```env
GROQ_API_KEY=your_groq_api_key_here
```

4. Make sure `ffmpeg` is available in `PATH`.
5. Run `SV5.exe`.

## Run From Source

```powershell
# Install the CUDA build of PyTorch first. If your NVIDIA driver does not support
# CUDA 12.6, choose a compatible command at https://pytorch.org/get-started/locally/.
python -m pip install --upgrade --force-reinstall torch==2.11.0+cu126 torchaudio==2.11.0+cu126 --index-url https://download.pytorch.org/whl/cu126
python -m pip install -r requirements.txt
python SV5.py
```

Create `.env` next to `SV5.py`:

```env
GROQ_API_KEY=your_groq_api_key_here
```

GigaAM does not require an API key. When **GigaAM v3 (local, GPU)** is selected,
the app checks that CUDA is available, downloads the model to `model_cache/gigaam/`
next to the application when needed, and loads it explicitly on `cuda`. It never
falls back to CPU: if CUDA PyTorch or an NVIDIA GPU is unavailable, the app returns
to Groq mode and shows an error.

## Build EXE

```powershell
python -m pip install pyinstaller
python -m PyInstaller --noconfirm --clean --onefile --windowed --name SV5 --icon ".\assets\ico.ico" --collect-all gigaam --collect-all hydra --collect-all omegaconf --collect-all torch --collect-all torchaudio .\SV5.py
```

The executable will be created at `dist\SV5.exe`.

## Notes

- `.env`, recorded audio, build folders, and packaged binaries are ignored by Git.
- Older project snapshots were imported into Git history with hardcoded API keys redacted.
- The project name intentionally follows the original requested spelling: `SallySpeechDesctop`.
