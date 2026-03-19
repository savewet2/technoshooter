import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
BOOKS_DIR = DATA_DIR / "books"
INDEX_DIR = DATA_DIR / "index"
TEMPLATES_DIR = BASE_DIR / "templates"

BOOKS_DIR.mkdir(parents=True, exist_ok=True)
INDEX_DIR.mkdir(parents=True, exist_ok=True)

# Ключ задавай в терминале:
# export GENAPI_API_KEY="твой_новый_ключ"
GENAPI_API_KEY = "sk-vxqwOVCM3RudJqOLK1akWIYZg0CABEinB5SQuZNyOrqrl5EiaZ69aWRhZdnW"

# OpenAI-compatible proxy GenAPI
GENAPI_BASE_URL = os.getenv("GENAPI_BASE_URL", "https://proxy.gen-api.ru/v1").strip()
GENAPI_MODEL = os.getenv("GENAPI_MODEL", "deepseek-chat").strip()

# Модель эмбеддингов для русского текста
EMBED_MODEL = os.getenv(
    "EMBED_MODEL",
    "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
).strip()

TOP_K = 5
MAX_CONTEXT_CHUNKS = 5

CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200

SEARCH_SCORE_THRESHOLD = 0.80

ALLOWED_EXTENSIONS = {".txt"}