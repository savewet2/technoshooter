from pathlib import Path
from typing import Dict, List
import re

from charset_normalizer import from_bytes

from app.config import CHUNK_OVERLAP, CHUNK_SIZE


def normalize_text(text: str) -> str:
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = text.replace("\x00", "")
    return text.strip()


def russian_score(text: str) -> float:
    if not text:
        return 0.0

    letters = re.findall(r"[А-Яа-яЁё]", text)
    bad = re.findall(r"[┼└┬▓░▒▀╚╩╨╥╤╧╬╠╣]", text)

    score = len(letters) - len(bad) * 2
    return score / max(len(text), 1)


def decode_bytes_safely(raw: bytes) -> str:
    candidates = []

    for enc in [
        "utf-8",
        "utf-8-sig",
        "cp1251",
        "windows-1251",
        "cp866",
        "koi8-r",
        "utf-16",
    ]:
        try:
            text = raw.decode(enc)
            candidates.append((enc, text, russian_score(text)))
        except Exception:
            pass

    try:
        best = from_bytes(raw).best()
        if best is not None:
            text = str(best)
            candidates.append(("charset-normalizer", text, russian_score(text)))
    except Exception:
        pass

    if not candidates:
        return raw.decode("utf-8", errors="replace")

    candidates.sort(key=lambda x: x[2], reverse=True)
    return candidates[0][1]


def split_into_chunks(text: str, source_name: str) -> List[Dict]:
    text = normalize_text(text)
    if not text:
        return []

    chunks: List[Dict] = []
    total_len = len(text)
    step = max(1, CHUNK_SIZE - CHUNK_OVERLAP)

    chunk_id = 0
    start = 0

    while start < total_len:
        end = min(start + CHUNK_SIZE, total_len)
        chunk_text = text[start:end].strip()

        if chunk_text:
            chunks.append(
                {
                    "text": chunk_text,
                    "source": source_name,
                    "chunk_id": chunk_id,
                    "char_start": start,
                    "char_end": end,
                    "relative_position_percent": int((start / max(total_len, 1)) * 100),
                }
            )
            chunk_id += 1

        if end >= total_len:
            break

        start += step

    return chunks


def load_book(file_path: Path) -> List[Dict]:
    raw = file_path.read_bytes()
    text = decode_bytes_safely(raw)
    return split_into_chunks(text, file_path.name)