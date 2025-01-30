import subprocess
import sys

def install_ffmpeg():
    print("Starting Ffmpeg installation...")
    
    subprocess.check_call([sys.executable, "-m", "pip", "install", "--upgrade", "pip"])
    subprocess.check_call([sys.executable, "-m", "pip", "install", "--upgrade", "setuptools"])
    
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "ffmpeg"])
        print("Installed ffmpeg-python successfully.")
    except subprocess.CalledProcessError as e:
        print("Failed to install ffmpeg-python via pip.")
    
    try:
        subprocess.check_call([
            "wget",
            "https://johnvansickle.com/ffmpeg/releases/ffmpeg-release-amd64-static.tar.xz",
            "-O",
            "/tmp/ffmpeg.tar.xz"
        ])
        
        subprocess.check_call([
            "tar", "-xf", "/tmp/ffmpeg.tar.xz", "-C", "/tmp/"
        ])
        
        result = subprocess.run(
            ["find", "/tmp", "-name", "ffmpeg", "-type", "f"],
            capture_output=True,
            text=True
        )
        ffmpeg_path = result.stdout.strip()
        