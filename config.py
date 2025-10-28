# config.py
import os
from dataclasses import dataclass
from dotenv import load_dotenv

load_dotenv()  # .env dosyasını okur

@dataclass(frozen=True)
class Config:
    # Dizinler
    ROOT_DIR: str = os.path.abspath(os.path.dirname(__file__))
    DATA_DIR: str = os.path.join(os.path.abspath(os.path.dirname(__file__)), "data")
    DOCS_DIR: str = os.path.join(os.path.abspath(os.path.dirname(__file__)), "data", "docs")
    CHROMA_DIR: str = os.path.join(
    os.path.abspath(os.path.dirname(__file__)),
    os.getenv("CHROMA_DIR", ".chroma")
    )

    # Gemini
    GEMINI_API_KEY: str = os.getenv("GEMINI_API_KEY", "")
    GEMINI_MODEL: str = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
    GEMINI_EMBED_MODEL: str = os.getenv("GEMINI_EMBED_MODEL", "gemini-embedding-001")

    # Uygulama
    TEMPERATURE: float = float(os.getenv("TEMPERATURE", "0.2"))

cfg = Config()

# Gerekli klasörleri garanti altına al
os.makedirs(cfg.DATA_DIR, exist_ok=True)
os.makedirs(cfg.DOCS_DIR, exist_ok=True)
os.makedirs(cfg.CHROMA_DIR, exist_ok=True)

def check_config() -> None:
    """Zorunlu değişkenleri doğrula."""
    missing = []
    if not cfg.GEMINI_API_KEY:
        missing.append("GEMINI_API_KEY")
    if missing:
        raise RuntimeError(
            f".env içinde eksik değişken(ler): {', '.join(missing)}. "
            "AI Studio'dan aldigin anahtari .env dosyasina yaz."
        )
