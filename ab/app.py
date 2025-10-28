# app.py
# — Ak-Koç — GenAI Destekli Genç Kart Koçu (Streamlit) —
import os
import traceback
import pandas as pd
import streamlit as st

from config import cfg
from rag import answer_question
from rules import detect_recurring, risk_check

# Opsiyonel: iç teşhis için
from retriever import get_collection, search
from ingest import ingest_docs

st.set_page_config(
    page_title="Ak-Koç — Genç Kart Koçu",
    page_icon="💳",
    layout="wide",
)

# ----------------------------- Yardımcılar -----------------------------
def _load_tx():
    path = os.path.join("data", "transactions.csv")
    if not os.path.exists(path):
        return None, path
    try:
        return pd.read_csv(path), path
    except Exception:
        return None, path

def _pill(text: str):
    st.markdown(
        f"""<span style="display:inline-block;padding:6px 10px;border-radius:999px;background:#F1F5F9;border:1px solid #E2E8F0;font-size:12px;margin-right:6px;">{text}</span>""",
        unsafe_allow_html=True,
    )

def _notice(msg: str, kind: str = "info"):
    if kind == "success": st.success(msg)
    elif kind == "warning": st.warning(msg)
    elif kind == "error": st.error(msg)
    else: st.info(msg)

# ----------------------------- Sidebar -----------------------------
st.sidebar.header("⚙️ Ayarlar")

top_k = st.sidebar.slider("Top K (bağlam parçası)", 1, 10, 5)
temperature = st.sidebar.slider("Cevap çeşitliliği (temperature)", 0.0, 1.0, 0.2, 0.1)

st.sidebar.markdown("---")
st.sidebar.caption("RAG kaynakları için `data/docs/` klasörüne PDF/TXT ekleyip **Reindex** basın.")

# Reindex butonu
if st.sidebar.button("🔄 İndeksi Tekrar Kur (Reindex)", use_container_width=True):
    with st.spinner("İndeks tekrar kuruluyor..."):
        try:
            added = ingest_docs(verbose=True)
            col = get_collection()
            _notice(f"İndeks güncellendi. Eklenen parça: **{added}** | Toplam kayıt: **{col.count()}**", "success")
        except Exception as e:
            st.sidebar.error("Reindex sırasında hata!")
            st.sidebar.code(traceback.format_exc())

# Hızlı durum
try:
    _col = get_collection()
    st.sidebar.markdown(f"**📊 Kayıt sayısı:** { _col.count() }")
except Exception:
    st.sidebar.markdown("**📊 Kayıt sayısı:** —")

# ----------------------------- Başlık -----------------------------
st.title("💳 Ak-Koç — GenAI Destekli Genç Kart Koçu")
st.caption("RAG + Koçluk Kuralları + Streamlit arayüz")

# ----------------------------- Sekmeler -----------------------------
tab_chat, tab_subs, tab_risk = st.tabs(["💬 Chatbot", "🔁 Abonelikler", "⚠️ Risk Koçluğu"])

# ============================= Chatbot =============================
with tab_chat:
    st.subheader("RAG Chatbot")

    # Basit chat state
    if "chat" not in st.session_state:
        st.session_state.chat = []

    # Mesajları göster
    for role, content in st.session_state.chat:
        with st.chat_message(role):
            st.markdown(content)

    # Kullanıcı girdisi
    prompt = st.chat_input("Sorunu yaz (örn: Ekstre nedir ve ödeme tarihi nasıl belirlenir?)")
    if prompt:
        # Kullanıcı mesajını ekle
        st.session_state.chat.append(("user", prompt))
        with st.chat_message("user"):
            st.markdown(prompt)

        # Model cevabı
        with st.chat_message("assistant"):
            with st.spinner("Yanıt üretiliyor..."):
                try:
                    resp = answer_question(prompt.strip(), top_k=top_k, temperature=temperature)
                except Exception:
                    st.error("Cevap üretirken hata oluştu.")
                    st.code(traceback.format_exc())
                    resp = {"answer": "Üzgünüm, şu an cevap üretilemiyor.", "sources": []}

            st.markdown("### 🧠 Yanıt")
            st.write(resp.get("answer", ""))

            # Kaynaklar
            sources = resp.get("sources") or []
            if sources:
                st.markdown("### 📚 Kaynaklar")
                for s in sources:
                    _pill(f"{s['title']} (chunk {s['chunk']})")
            else:
                st.info("Kaynak bulunamadı. Muhtemel nedenler: indeks boş veya farklı dizin. Yan menüden **Reindex** deneyin.")

        # Cevabı state’e ekle
        st.session_state.chat.append(("assistant", resp.get("answer", "")))

    with st.expander("🔧 Tanı / Debug"):
        try:
            col = get_collection()
            st.write("Kayıt sayısı:", col.count())
            sample = search("Ekstre nedir?", k=2)
            st.write("Örnek arama:", [
                {"title": h["meta"].get("title"), "chunk": h["meta"].get("chunk")}
                for h in sample
            ])
            st.write("DOCS_DIR:", cfg.DOCS_DIR)
            st.write("CHROMA_DIR:", cfg.CHROMA_DIR)
        except Exception:
            st.write("Tanı sırasında hata:")
            st.code(traceback.format_exc())

# ============================ Abonelikler ============================
with tab_subs:
    st.subheader("Hayalet Abonelik Tespiti")

    df, tx_path = _load_tx()
    if df is None:
        st.warning(f"`{tx_path}` bulunamadı. Sol menüden Reindex bağımsızdır; abonelik verisi için `simulator.py` çalıştırın.")
        if st.button("Örnek veri oluştur (simulator.py)"):
            os.system("python simulator.py")
            st.rerun()
    else:
        st.caption(f"Dosya: `{tx_path}`")
        st.dataframe(df.tail(25), width="stretch")  # <- deprecation fix: width='stretch'
        subs = []
        try:
            subs = detect_recurring(df, days_ahead=7)
        except Exception:
            st.error("Abonelik tespitinde hata:")
            st.code(traceback.format_exc())
        if subs:
            st.success("Yaklaşan abonelik yenilemeleri:")
            for s in subs:
                _pill(f"{s['merchant']} — {s['amount']}₺ — {s['next_renewal']}")
        else:
            st.info("Yaklaşan abonelik yenilemesi bulunamadı.")

# ============================ Risk Koçluğu ============================
with tab_risk:
    st.subheader("Risk Kuralları")
    df, tx_path = _load_tx()
    if df is None:
        st.warning(f"`{tx_path}` bulunamadı. `python simulator.py` ile oluşturabilirsiniz.")
    else:
        try:
            alerts = risk_check(df)
            if alerts:
                for a in alerts:
                    st.error(a)
            else:
                st.success("Riskli bir durum tespit edilmedi. Böyle devam! ✅")
        except Exception:
            st.error("Risk kontrolünde hata:")
            st.code(traceback.format_exc())
