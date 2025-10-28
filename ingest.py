# ingest.py
# --- Akbank GenAI Bootcamp: Belgeleri indeksleme ---
from __future__ import annotations
import os, re, time, traceback
from typing import List, Dict

from config import cfg
from embedder import Embedder               # <-- DÜZELTME: Embedder sınıfını kullanıyoruz
from retriever import get_collection, index_texts

try:
    from pypdf import PdfReader            # PDF için
    HAS_PYPDF = True
except Exception:
    HAS_PYPDF = False

# ---- Ayarlar ----
CHUNK_SIZE = 800
CHUNK_OVERLAP = 120
BATCH_SIZE = 16
MAX_CHUNKS = 200

def _read_txt(path: str) -> str:
    # Windows'ta encoding sorunu yaşamamak için birkaç deneme
    for enc in ("utf-8", "utf-8-sig", "latin-1"):
        try:
            with open(path, "r", encoding=enc, errors="strict") as f:
                return f.read()
        except UnicodeDecodeError:
            continue
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        return f.read()

def _read_pdf(path: str) -> str:
    if not HAS_PYPDF:
        raise RuntimeError("pypdf kurulu değil. `pip install pypdf` ile kurun veya TXT kullanın.")
    reader = PdfReader(path)
    pages = []
    for p in reader.pages:
        pages.append(p.extract_text() or "")
    return "\n".join(pages)

def _normalize(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "")).strip()

def _chunk(text: str, size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> List[str]:
    out, start, L = [], 0, len(text)
    if L == 0: return out
    while start < L:
        end = min(L, start + size)
        out.append(text[start:end])
        start = max(0, end - overlap)
        if start >= L: break
    return out

def _scan_docs() -> List[str]:
    os.makedirs(cfg.DOCS_DIR, exist_ok=True)
    return sorted(
        os.path.join(cfg.DOCS_DIR, n)
        for n in os.listdir(cfg.DOCS_DIR)
        if n.lower().endswith((".txt", ".pdf"))
    )

def _ensure_demo(files: List[str]) -> List[str]:
    if files:
        return files
    demo = os.path.join(cfg.DOCS_DIR, "akbank_demo.txt")
    with open(demo, "w", encoding="utf-8") as f:
        f.write(
            "Ekstre, kredi kartı dönemindeki harcamaları gösteren belgedir. "
            "Ödeme tarihi ekstre kesim tarihinden sonra belirlenir. "
            "Kazanılan Harçlık (bonus) kampanyalara göre kullanılır."
        )
    print(f"ℹ️ Demo dosyası oluşturuldu: {demo}")
    return [demo]

def ingest_docs(verbose: bool = True) -> int:
    t0 = time.time()
    try:
        if verbose:
            print("🚀 ingest.py başladı")
            print("📂 DOCS_DIR   :", cfg.DOCS_DIR)
            print("💾 CHROMA_DIR :", cfg.CHROMA_DIR)

        # Chroma koleksiyonunu hazırla
        col = get_collection()

        # Dosyaları topla (yoksa demo yarat)
        files = _ensure_demo(_scan_docs())
        if verbose:
            print(f"🔎 Bulunan dosyalar ({len(files)}): {[os.path.basename(x) for x in files]}")

        all_chunks: List[str] = []
        metadatas: List[Dict] = []
        ids: List[str] = []

        # Dosya okuma + parçalama
        for path in files:
            ext = os.path.splitext(path)[1].lower()
            print(f"\n📄 İşleniyor: {os.path.basename(path)} ({ext})", flush=True)
            raw = _read_pdf(path) if ext == ".pdf" else _read_txt(path)
            print("   - Okunan karakter:", len(raw))
            text = _normalize(raw)
            chunks = _chunk(text)
            print("   - Normalleşmiş:", len(text), "| Chunk:", len(chunks))

            base = os.path.basename(path)
            title = os.path.splitext(base)[0]
            for i, ch in enumerate(chunks):
                all_chunks.append(ch)
                metadatas.append({"source": base, "title": title, "chunk": i})
                ids.append(f"{title}-{i}")

        if not all_chunks:
            print("⚠️ Parçalanacak veri yok.")
            return 0

        # (opsiyonel) güvenli üst sınır
        if len(all_chunks) > MAX_CHUNKS:
            all_chunks = all_chunks[:MAX_CHUNKS]
            metadatas = metadatas[:MAX_CHUNKS]
            ids = ids[:MAX_CHUNKS]
            print(f"ℹ️ Geçici limit: ilk {MAX_CHUNKS} parça işlenecek.")

        # Embedding üret
        print(f"\n🧩 Embedding başlıyor — toplam parça: {len(all_chunks)} (batch={BATCH_SIZE})", flush=True)
        emb = Embedder(batch_size=BATCH_SIZE)
        t1 = time.time()
        vectors = emb.embed_batched(all_chunks)
        t2 = time.time()
        dim = len(vectors[0]) if vectors else 0
        print(f"✅ Embedding tamam: {len(vectors)} vektör | boyut={dim} | süre={t2 - t1:.1f}s")

        # Chroma'ya yaz
        print("🗂️ Chroma'ya yazılıyor…")
        added = index_texts(all_chunks, metadatas, ids, vectors)
        print(f"🎉 İndekse eklenen: {added}")

        try:
            print("📊 Koleksiyon toplam kayıt:", col.count())
        except Exception:
            pass

        print(f"🏁 ingest.py bitti | toplam süre: {time.time() - t0:.1f}s")
        return added

    except Exception as e:
        print("❌ ingest.py hata yakalandı!\n" + traceback.format_exc())
        return 0

if __name__ == "__main__":
    ingest_docs(verbose=True)
