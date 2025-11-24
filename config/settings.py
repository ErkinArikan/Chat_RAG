import os

# Dosya yolları
DATA_DIR = "data"
DOC_PATH = os.path.join(DATA_DIR, "document.txt")
INDEX_PATH = os.path.join(DATA_DIR, "rag_index.pkl")

# Modeller
EMBED_MODEL = "text-embedding-3-small"
CHAT_MODEL = "gpt-5-nano"

# RAG parametreleri
CHUNK_SIZE = 400        # 800 yerine daha küçük, daha odaklı
TOP_K = 3               # default (fallback)
MAX_TOP_K = 5           # dinamik üst sınır
SIM_THRESHOLD = 0.30    # bu skorun altındakileri mümkünse alma
OVERLAP_SENTENCES = 1   # bir önceki chunk'tan kaç cümle taşıyalım