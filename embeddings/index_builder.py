import os
import textwrap
import pickle
import numpy as np

from config import settings
from services.openai_client import client

# NLTK ile cümle bazlı split (varsa)
try:
    import nltk
    from nltk.tokenize import sent_tokenize
    _NLTK_AVAILABLE = True
except ImportError:
    _NLTK_AVAILABLE = False
    print("[WARN] nltk bulunamadı, basit cümle bölme kullanılacak.")
    

def load_document(path: str) -> str:
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


def split_paragraphs(text: str) -> list[str]:
    """
    Boş satırlara göre paragraf bazlı bölüyoruz.
    """
    paragraphs = []
    current = []

    for line in text.splitlines():
        if line.strip() == "":
            if current:
                paragraphs.append(" ".join(current).strip())
                current = []
        else:
            current.append(line.strip())
    if current:
        paragraphs.append(" ".join(current).strip())

    return [p for p in paragraphs if p]


def split_sentences(paragraph: str) -> list[str]:
    """
    Paragrafı cümlelere böler (önce NLTK, yoksa basit fallback).
    """
    if _NLTK_AVAILABLE:
        return [s.strip() for s in sent_tokenize(paragraph) if s.strip()]
    else:
        # Çok basit fallback: . ! ? karakterlerine göre kaba split
        rough = textwrap.wrap(paragraph, width=len(paragraph))  # tek elemanlı liste
        text = rough[0] if rough else paragraph
        for ch in ["?", "!", "."]:
            text = text.replace(ch, f"{ch}<SPLIT>")
        parts = [p.strip() for p in text.split("<SPLIT>") if p.strip()]
        return parts


def semantic_chunks(text: str, max_chars: int,
                    overlap_sentences: int = 1) -> list[str]:
    """
    LlamaIndex tarzı bir yaklaşımın sade hali:
    - Önce paragraf bazlı böl
    - Her paragrafı cümlelere böl
    - Cümleleri max_chars sınırını geçmeyecek şekilde grupla
    - Her chunk, bir önceki chunk'tan overlap_sentences kadar cümle taşır (overlap)
    """
    paragraphs = split_paragraphs(text)
    chunks: list[str] = []

    for para in paragraphs:
        sentences = split_sentences(para)
        if not sentences:
            continue

        current_sentences: list[str] = []
        current_len = 0

        for sent in sentences:
            sent_len = len(sent)

            # Bu cümleyi eklersek max_chars aşılacaksa → yeni chunk başlat
            if current_sentences and (current_len + sent_len > max_chars):
                chunk_text = " ".join(current_sentences).strip()
                if chunk_text:
                    chunks.append(chunk_text)

                # Overlap: son N cümleyi yeni chunk'ın başlangıcına taşı
                if overlap_sentences > 0:
                    current_sentences = current_sentences[-overlap_sentences:]
                    current_len = sum(len(s) for s in current_sentences)
                else:
                    current_sentences = []
                    current_len = 0

            # Cümleyi mevcut chunk'a ekle
            current_sentences.append(sent)
            current_len += sent_len

        # Paragraf sonu, elde kalanları da ekle
        if current_sentences:
            chunk_text = " ".join(current_sentences).strip()
            if chunk_text:
                chunks.append(chunk_text)

    # Son temizlik
    return [c for c in chunks if c.strip()]


def embed_texts(texts: list[str]) -> np.ndarray:
    resp = client.embeddings.create(
        model=settings.EMBED_MODEL,
        input=texts,
    )
    vectors = [np.array(item.embedding, dtype="float32") for item in resp.data]
    return np.vstack(vectors)  # (N, D)


def build_index():
    os.makedirs(settings.DATA_DIR, exist_ok=True)

    if not os.path.exists(settings.DOC_PATH):
        raise FileNotFoundError(f"Doküman bulunamadı: {settings.DOC_PATH}")

    print(f"📄 Doküman yükleniyor: {settings.DOC_PATH}")
    text = load_document(settings.DOC_PATH)

    print("✂️ Semantic chunking başlıyor (paragraf + cümle + overlap)...")
    chunks = semantic_chunks(
        text,
        max_chars=settings.CHUNK_SIZE,
        overlap_sentences=getattr(settings, "OVERLAP_SENTENCES", 1),
    )
    print(f"➡️ Toplam {len(chunks)} adet semantic chunk oluşturuldu.")

    print("🧠 Chunk'lar embed ediliyor...")
    embeddings = embed_texts(chunks)

    data = {
        "texts": chunks,
        "embeddings": embeddings,
        "model": settings.EMBED_MODEL,
    }

    print(f"💾 Index kaydediliyor: {settings.INDEX_PATH}")
    with open(settings.INDEX_PATH, "wb") as f:
        pickle.dump(data, f)

    print("✅ Tamamlandı.")