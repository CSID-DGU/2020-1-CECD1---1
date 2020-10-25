import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SERVER_DIR = os.path.join(BASE_DIR, "server")


def audio_resolution(original_audio):
    with open(f"{SERVER_DIR}/src/file.txt", "w+") as f:
        f.write(original_audio)
    os.system(
        f"python {SERVER_DIR}/src/run.py eval --logname {SERVER_DIR}/src/model.ckpt --wav-file-list {SERVER_DIR}/src/file.txt --r 4"
    )
