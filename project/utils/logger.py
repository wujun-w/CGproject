from datetime import datetime
from pathlib import Path


class SimpleLogger:
    def __init__(self, log_path):
        self.log_path = Path(log_path)
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        self.handle = self.log_path.open("a", encoding="utf-8")

    def log(self, message):
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        text = f"[{timestamp}] {message}"
        print(text, flush=True)
        self.handle.write(text + "\n")
        self.handle.flush()

    def close(self):
        self.handle.close()
