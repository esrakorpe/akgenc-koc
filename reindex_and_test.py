# reindex_and_test.py
from retriever import reset_collection, get_collection, search
from ingest import ingest_docs
from config import cfg
import os

print("📌 CHROMA_DIR:", cfg.CHROMA_DIR)
print("📌 DOCS_DIR  :", cfg.DOCS_DIR)
os.makedirs(cfg.DOCS_DIR, exist_ok=True)
print("📁 DOCS:", os.listdir(cfg.DOCS_DIR))

print("\n🧹 Koleksiyon sıfırlanıyor...")
reset_collection()
print("✅ Sıfırlandı.")

print("\n📥 ingest_docs() başlıyor…")
added = ingest_docs(verbose=True)
print("✅ ingest_docs() bitti — eklenen:", added)

col = get_collection()
print("\n📊 Koleksiyon toplam kayıt:", col.count())

print("\n🔎 Örnek arama: 'Ekstre nedir?'")
hits = search("Ekstre nedir?", k=3)
for h in hits:
    print(" -", h["meta"], "|", (h["doc"] or "")[:90], "…")
print("\n[SON] Reindex & test tamam.")
