# 💬 Chat_RAG – Şirket İçi Doküman Asistanı

Bu proje, **şirket içi dokümanlara dayalı** çalışan destek sistemi örneğidir.  
Amaç: Çalışan, web arayüzündeki chat kutusuna soru yazar → sistem RAG (Retrieval-Augmented Generation) kullanarak dokümandan ilgili parçayı bulur → OpenAI modeliyle bu parçaya dayanarak cevap üretir.

---

## 🧠 Mimari Özet

Akış kabaca şöyle:

1. `data/document.txt` içindeki büyük metin, **semantic chunk**’lara bölünüyor.
2. Her chunk için **embedding** üretiliyor (`text-embedding-3-small`).
3. Bu vektörler ve chunk metinleri tek bir dosyada (`data/rag_index.pkl`) saklanıyor.
4. FastAPI backend’i çalışırken:
   - Kullanıcının sorusunu embed ediyor,
   - En benzer chunk’ları buluyor (cosine similarity),
   - Bu chunk’ları ve soruyu GPT modeline (`gpt-5-nano` gibi) gönderiyor,
   - Gelen cevabı chat arayüzüne geri dönüyor.
5. Frontend tarafında küçük bir **chat widget** (`static/chat.html`) var; backend’e `/ask` endpoint’ine istek atıyor.

---

## 📂 Klasör / Dosya Yapısı

Örnek proje yapısı (seninkine çok yakın):

```text
company_rag/
├── app.py                  # FastAPI giriş noktası (uvicorn buradan çalışıyor)
├── build_index.py          # Dokümandan RAG index (embedding) oluşturan script
├── requirements.txt        # Python bağımlılıkları
├── .env                    # OPENAI_API_KEY burada (gitignore’da)
├── .gitignore              # venv, .env, pkl vb. ignore
│
├── data/
│   ├── document.txt        # Şirket içi doküman (metin)
│   └── rag_index.pkl       # Embedding + chunk bilgilerini tutan vektör index
│
├── config/
│   └── settings.py         # Model isimleri, dosya yolları, RAG ayarları
│
├── services/
│   └── openai_client.py    # OpenAI client’ı, API key okuma vs.
│
├── embeddings/
│   └── index_builder.py    # Semantic chunking + embedding üretimi
│
├── rag/
│   ├── retriever.py        # Soru → embedding → benzer chunk’ları bulma
│   └── pipeline.py         # RAG pipeline: retriever + GPT cevabı
│
├── api/
│   └── routes.py           # FastAPI router: /ask endpoint’i, request/response modelleri
│
└── static/
    └── chat.html           # Floating chat widget (frontend)
