import os
import pickle
import numpy as np

from config import settings
from services.openai_client import client
from rag.similarity import cosine_sim

# Index yükle
if not os.path.exists(settings.INDEX_PATH):
    raise FileNotFoundError(
        f"Index dosyası bulunamadı: {settings.INDEX_PATH}\n"
        f"Önce 'python build_index.py' çalıştırdığından emin ol."
    )

with open(settings.INDEX_PATH, "rb") as f:
    index_data = pickle.load(f)

DOC_TEXTS = index_data["texts"]        # List[str]
DOC_EMB = index_data["embeddings"]     # np.array (N, D)


def get_relevant_chunks(question: str):
    """
    - Soruyu embed eder
    - Tüm doküman embedding’leri ile cosine similarity hesaplar
    - similarity skoruna göre:
        * threshold üstündekileri al
        * en fazla MAX_TOP_K
        * hiçbir şey yoksa en iyi 1 chunk'ı al (fallback)
    """
    emb = client.embeddings.create(
        model=settings.EMBED_MODEL,
        input=[question],
    )

    q_vec = np.array(emb.data[0].embedding, dtype="float32")
    sims = cosine_sim(q_vec, DOC_EMB)  # (N,)

    # Skorları (index, skor) olarak sırala
    sorted_idx = np.argsort(sims)[::-1]  # büyükten küçüğe
    sorted_scores = sims[sorted_idx]

    # Debug: ilk 10 skoru göster
    print("\n[DEBUG] Similarity top 10:")
    for i in range(min(10, len(sorted_idx))):
        print(f"  #{i+1}: idx={sorted_idx[i]}, score={sorted_scores[i]:.4f}")

    max_top_k = getattr(settings, "MAX_TOP_K", settings.TOP_K)
    threshold = getattr(settings, "SIM_THRESHOLD", 0.0)

    selected_indices = []

    # Threshold üstünde olanları ekle
    for idx, score in zip(sorted_idx, sorted_scores):
        if score < threshold:
            break
        selected_indices.append(idx)
        if len(selected_indices) >= max_top_k:
            break

    # Eğer threshold'u aşan yoksa → en iyi 1 taneyi yine de al
    if not selected_indices:
        best_idx = int(sorted_idx[0])
        print(f"[DEBUG] Threshold üstünde sonuç yok, fallback olarak idx={best_idx} alındı.")
        selected_indices = [best_idx]

    chunks = [DOC_TEXTS[i] for i in selected_indices]

    print(f"[DEBUG] Seçilen chunk sayısı: {len(chunks)}")
    return chunks