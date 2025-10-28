# rag.py
# ——— RAG: retrieval -> context build -> Gemini yanıtı ———
from __future__ import annotations
from typing import List, Dict, Any, Optional
import textwrap, html, sys

from config import cfg                  # TEMPERATURE vb.
from retriever import search as rsearch # Chroma araması
from gemini_client import generate_text # Gemini çağrısı

SYSTEM = textwrap.dedent("""
Sen 'Ak-Koç' isimli finansal yardımcı bir asistansın.
- Ekstre, ödeme tarihi, limit ve bonus/harçlık konularında kısa ve öğretici cevap ver.
- Yalnızca verilen BAĞLAM içindeki bilgilere güven; yoksa "Bilmiyorum" de.
- Cevabın sonunda 'Kaynaklar:' başlığında kullandığın parça başlıklarını listele.
""").strip()

def _truncate(s: str, n: int = 1200) -> str:
    s = (s or "").strip()
    return s if len(s) <= n else (s[:n-1].rstrip() + "…")

def _format_context(hits: List[Dict[str, Any]], max_chars: int = 4000) -> str:
    """Hit’leri numaralı bloklar halinde birleştirir; toplam uzunluğu sınırlar."""
    out, total = [], 0
    for i, h in enumerate(hits, 1):
        title = h["meta"].get("title") or h["meta"].get("source") or "dokuman"
        chunk = h["meta"].get("chunk", 0)
        snippet = _truncate(h.get("doc", ""), 1200)
        block = f"[{i}] {title} (chunk {chunk}):\n{snippet}\n"
        if total + len(block) > max_chars:
            break
        out.append(block); total += len(block)
    return "\n".join(out)

def _format_sources(hits: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Kaynakları tekilleştirip {title, source, chunk} listesi döndürür."""
    seen, out = set(), []
    for h in hits:
        t = h["meta"].get("title") or h["meta"].get("source") or "dokuman"
        s = h["meta"].get("source", ""); c = h["meta"].get("chunk", 0)
        key = (t, s, c)
        if key in seen: continue
        seen.add(key); out.append({"title": t, "source": s, "chunk": c})
    return out

def answer_question(
    query: str,
    top_k: int = 5,
    temperature: Optional[float] = None,
    max_context_chars: int = 4000,
) -> Dict[str, Any]:
    """Sorguyu alır, bağlamı toplayıp Gemini ile cevap üretir."""
    print("⏩ [RAG] retrieval başlıyor...", flush=True)
    hits = rsearch(query, k=top_k)
    print(f"⏹️  [RAG] retrieval bitti — hit sayısı: {len(hits)}", flush=True)

    if not hits:
        return {
            "answer": "Bilmiyorum. Şu an için ilgili bir kaynak bulamadım.",
            "sources": [],
            "meta": {"used_hits": 0, "top_k": top_k, "context_chars": 0},
        }

    ctx = html.unescape(_format_context(hits, max_chars=max_context_chars))
    print(f"🧩 [RAG] bağlam hazır — uzunluk: {len(ctx)}", flush=True)

    prompt = textwrap.dedent(f"""
    {SYSTEM}

    --- BAĞLAM BAŞI ---
    {ctx}
    --- BAĞLAM SONU ---

    SORU:
    {query}

    Lütfen kısa ve net bir cevap ver. Emin değilsen "Bilmiyorum" de.
    Sonunda 'Kaynaklar:' başlığı altında kullandığın parçaların başlıklarını listele.
    """)

    temp = cfg.TEMPERATURE if temperature is None else temperature
    print("🤖 [RAG] Gemini çağrısı başlıyor...", flush=True)
    raw = generate_text(prompt, temperature=temp)
    print("✅ [RAG] Gemini yanıtı alındı.", flush=True)

    answer = (raw or "").strip() or "Bilmiyorum. Şu an için ilgili bir kaynak bulamadım."
    return {
        "answer": answer,
        "sources": _format_sources(hits),
        "meta": {"used_hits": len(hits), "top_k": top_k, "context_chars": len(ctx)},
    }

# ——— CLI testi ———
if __name__ == "__main__":
    print("🔎 RAG test — örnek sorgu ile çalıştırılıyor", flush=True)
    q = "Ekstre nedir ve ödeme tarihi nasıl belirlenir?"
    resp = answer_question(q, top_k=5)
    print("\n🧠 Yanıt:\n", resp["answer"], flush=True)
    print("\n📚 Kaynaklar:", resp["sources"], flush=True)
    print("\n⚙️ Meta:", resp.get("meta", {}), flush=True)
