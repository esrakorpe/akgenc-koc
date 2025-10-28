# rules.py
import pandas as pd
from datetime import datetime, timedelta

def _as_date(s):
    return pd.to_datetime(s).date()

def detect_recurring(df: pd.DataFrame, days_ahead: int = 7):
    """
    Basit hayalet abonelik tespiti:
    - Aynı merchant adına ~30 günde bir tekrar eden ve benzer tutarlı işlemler
    - Son işlem + 30 gün ≈ bir sonraki yenileme; yenileme days_ahead gün içindeyse bildir
    """
    subs = []
    if df.empty: return subs
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"]).dt.date
    today = datetime.now().date()

    for name, g in df.groupby("merchant"):
        g = g.sort_values("date")
        if len(g) < 3:  # en az 3 tekrar
            continue
        # benzer tutar kontrolü: en çok geçen tutarı moda al
        top_amount = g["amount"].round(1).mode()
        if top_amount.empty:
            continue
        top_amount = float(top_amount.iloc[0])
        # kabaca 30 günlük periyot say
        last = g["date"].iloc[-1]
        next_renewal = last + timedelta(days=30)
        if 0 <= (next_renewal - today).days <= days_ahead:
            subs.append({
                "merchant": name,
                "amount": top_amount,
                "last_date": last,
                "next_renewal": next_renewal
            })
    return subs

def risk_check(df: pd.DataFrame):
    """
    Basit risk kuralları:
    - Son 3 günde toplam harcama > 3000 TL
    - Art arda 3 yüksek tutar (> 1000 TL)
    """
    alerts = []
    if df.empty: return alerts
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"]).dt.date
    df = df.sort_values("date")

    # Kural 1: son 3 gün toplamı
    last3 = df[df["date"] >= (datetime.now().date() - timedelta(days=3))]
    if last3["amount"].sum() > 3000:
        alerts.append("Son 3 günde toplam harcama 3000₺ üzeri – Kart Koçun Uyardı! 🚨")

    # Kural 2: art arda 3 yüksek harcama
    highs = (df["amount"] > 1000).astype(int).tolist()
    streak = 0
    for x in highs:
        streak = streak + 1 if x == 1 else 0
        if streak >= 3:
            alerts.append("Arka arkaya yüksek tutarlı harcamalar tespit edildi – Cüzdanı yavaşlatma zamanı olabilir.")
            break

    return alerts

if __name__ == "__main__":
    import os
    tx_path = os.path.join("data", "transactions.csv")
    if not os.path.exists(tx_path):
        print("ℹ️ data/transactions.csv bulunamadı. Önce simulator.py çalıştırın.")
    else:
        df = pd.read_csv(tx_path)
        print("🔁 Abonelikler:", detect_recurring(df, days_ahead=7))
        print("⚠️ Risk Uyarıları:", risk_check(df))
