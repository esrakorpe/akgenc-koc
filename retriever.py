# retriever.py
from typing import List, Dict, Any
from chromadb import PersistentClient
from config import cfg
from gemini_client import embed_query

COLLECTION_NAME = "ak_koc_docs"

def _client() -> PersistentClient:
    # Kalıcı Chroma istemcisi (.chroma klasöründe dosyalar)
    return PersistentClient(path=cfg.CHROMA_DIR)

def get_collection():
    client = _client()
    # cosine benzerlik (HNSW vektör uzayı)
    return client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"}
    )

def index_texts(
    texts: List[str],
    metadatas: List[Dict[str, Any]],
    ids: List[str],
    embeddings: List[List[float]],
) -> int:
    """
    Verilen metin/embedding/metadata listelerini koleksiyona ekler.
    """
    if not (len(texts) == len(metadatas) == len(ids) == len(embeddings)):
        raise ValueError("texts/metadatas/ids/embeddings uzunlukları eşleşmiyor.")
    col = get_collection()
    col.add(documents=texts, metadatas=metadatas, ids=ids, embeddings=embeddings)
    return len(ids)

def search(query: str, k: int = 5):
    """
    Sorguyu embed edip en yakın k dokümanı döndürür.
    """
    col = get_collection()
    qvec = embed_query(query)
    res = col.query(
        query_embeddings=[qvec],
        n_results=k,
        include=["documents", "metadatas", "distances"],
    )
    hits = []
    # Sonuç yapısı listelerin listesi şeklinde döner
    if res["ids"]:
        for i in range(len(res["ids"][0])):
            hits.append({
                "id": res["ids"][0][i],
                "doc": res["documents"][0][i],
                "meta": res["metadatas"][0][i],
                "distance": res["distances"][0][i],
            })
    return hits

def count() -> int:
    return get_collection().count()

def reset_collection():
    """
    Koleksiyonu sıfırlar (dikkat: tüm veriler silinir!).
    Geliştirme/test amaçlı.
    """
    client = _client()
    try:
        client.delete_collection(COLLECTION_NAME)
    except Exception:
        pass
    return get_collection()

# --------- Hızlı test ---------
if __name__ == "__main__":
    col = get_collection()
    print("✅ retriever hazır; mevcut kayıt sayısı:", col.count())
    demo = search("Ekstre nedir?", k=1)
    print("🔎 örnek arama (boş olabilir):", demo)
