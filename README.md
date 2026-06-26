# SallySpeechDesctop

SallySpeechDesctop is a small Windows desktop dictation tool. Press a global hotkey, speak, press it again, and the recognized Russian text is inserted into the active application.

The current build uses Groq's Whisper API (`whisper-large-v3`) and a PyQt6 desktop window.

## Features

- Global recording hotkey: `Ctrl+Alt+Space`
- Exit hotkey: `End`
- Groq Whisper transcription
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
python -m pip install -r requirements.txt
python SV5.py
```

Create `.env` next to `SV5.py`:

```env
GROQ_API_KEY=your_groq_api_key_here
```

## Build EXE

```powershell
python -m pip install pyinstaller
python -m PyInstaller --noconfirm --clean --onefile --windowed --name SV5 --icon ".\assets\ico.ico" .\SV5.py
```

The executable will be created at `dist\SV5.exe`.

## Notes

- `.env`, recorded audio, build folders, and packaged binaries are ignored by Git.
- Older project snapshots were imported into Git history with hardcoded API keys redacted.
- The project name intentionally follows the original requested spelling: `SallySpeechDesctop`.
