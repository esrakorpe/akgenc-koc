# gemini_client.py  — test çıktılı sürüm
import os, time
from typing import List, Sequence
from dotenv import load_dotenv
import google.generativeai as genai

# .env dosyasını oku
load_dotenv()

# Ortam değişkenlerini al
API_KEY = os.getenv("GEMINI_API_KEY", "")
GEN_MODEL = os.getenv("GEMINI_MODEL", "gemini-1.5-flash")
EMB_MODEL = os.getenv("GEMINI_EMBED_MODEL", "models/embedding-001")
TEMP = float(os.getenv("TEMPERATURE", "0.2"))

# Anahtar kontrolü
if not API_KEY:
    raise RuntimeError("GEMINI_API_KEY .env içinde bulunamadı — anahtarı ekle!")

# Gemini istemcisini yapılandır
genai.configure(api_key=API_KEY)

# -------- Metin Üretimi --------
def generate_text(prompt: str, temperature: float | None = None, system: str = "") -> str:
    """
    Gemini modelinden kısa bir yanıt alır.
    """
    temperature = TEMP if temperature is None else temperature
    model = genai.GenerativeModel(GEN_MODEL)
    contents = f"{system}\n\n{prompt}" if system else prompt
    resp = model.generate_content(contents, generation_config={"temperature": temperature})
    return (resp.text or "").strip()

# -------- Embedding (RAG için) --------
def embed_texts(texts: Sequence[str], sleep: float = 0.02) -> List[List[float]]:
    """
    Metin listesini embedding vektörlerine çevirir.
    """
    vecs: List[List[float]] = []
    for t in texts:
        r = genai.embed_content(model=EMB_MODEL, content=t)
        vecs.append(r["embedding"])
        if sleep:
            time.sleep(sleep)
    return vecs

def embed_query(text: str) -> List[float]:
    """
    Tek bir sorgu cümlesi için embedding döndürür.
    """
    r = genai.embed_content(model=EMB_MODEL, content=text)
    return r["embedding"]

# -------- Test Bloğu --------
if __name__ == "__main__":
    print("🔍 Gemini API bağlantısı test ediliyor...")
    try:
        # Basit text üretim testi
        yanit = generate_text("Merhaba! 1 cümlede kendini tanıt.")
        print("🧠 Model Yanıtı:", yanit)

        # Embedding testi
        emb = embed_texts(["ekstre nedir?"])
        print("📈 Embedding boyutu:", len(emb[0]))

        print("✅ Test başarıyla tamamlandı.")
    except Exception as e:
        print("❌ Hata oluştu:", e)
