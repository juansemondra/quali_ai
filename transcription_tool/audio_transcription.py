import subprocess
import os

audio_path = "/Users/juansemondra/Desktop/Archivos/Py Projects/AT/audio.mp3"  
output_path = os.path.splitext(audio_path)[0] + ".txt"  

subprocess.run([
    "whisper", audio_path, 
    "--language", "es",  
    "--output_format", "txt",  
    "--output_dir", os.path.dirname(audio_path)  
])

print(f"Transcripción guardada en {output_path}")