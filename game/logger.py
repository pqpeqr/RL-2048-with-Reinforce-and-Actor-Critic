from datetime import datetime
from typing import Optional




def log_append(message: str, filepath: str = "game2048.log", *, ts: Optional[datetime] = None) -> None:
    t = (ts or datetime.now()).isoformat(timespec="seconds")
    safe = message.replace("\n", " ")
    with open(filepath, "a", encoding="utf-8") as f:
        f.write(f"[{t}] {safe}\n")