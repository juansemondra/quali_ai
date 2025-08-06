#!/usr/bin/env python3
# transcribe.py

import os
import subprocess
import sys

# Directorios de entrada y salida
AUDIO_DIR = "audio_data/audio"
OUT_DIR   = "audio_data/transcriptions"

# Modelo Whisper a usar (p.ej. tiny, base, small, medium, large)
WHISPER_MODEL = "small"

def transcribe_file(audio_path: str, out_dir: str) -> None:
    """
    Llama al CLI `whisper` para transcribir `audio_path`,
    guardando el .txt resultante en `out_dir`.
    """
    cmd = [
        "whisper",
        audio_path,
        "--model", WHISPER_MODEL,
        "--output_dir", out_dir,
        "--output_format", "txt",
        "--no_speech_threshold", "0.5"
    ]
    try:
        subprocess.run(cmd, check=True, capture_output=True, text=True)
        print(f"✔ Transcribed: {os.path.basename(audio_path)}")
    except subprocess.CalledProcessError as e:
        print(f"✖ Failed: {os.path.basename(audio_path)}\n{e.stderr}", file=sys.stderr)

def main():
    if not os.path.isdir(AUDIO_DIR):
        print(f"Error: no existe el directorio de audio `{AUDIO_DIR}`.", file=sys.stderr)
        sys.exit(1)

    os.makedirs(OUT_DIR, exist_ok=True)

    # Extensiones de audio a procesar
    exts = {".wav", ".mp3", ".m4a", ".flac", ".aac", ".ogg"}
    files = [
        os.path.join(AUDIO_DIR, f)
        for f in sorted(os.listdir(AUDIO_DIR))
        if os.path.splitext(f.lower())[1] in exts
    ]

    if not files:
        print(f"⚠️ No se encontraron archivos de audio en `{AUDIO_DIR}`.")
        return

    for audio in files:
        transcribe_file(audio, OUT_DIR)

    print(f"\n✅ Transcripción completada. Archivos .txt en `{OUT_DIR}`.")

if __name__ == "__main__":
    main()