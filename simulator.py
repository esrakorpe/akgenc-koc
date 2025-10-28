# simulator.py
import os, random
import pandas as pd
from datetime import datetime, timedelta

random.seed(42)

USERS = ["ogrenci1@uni.edu"]
SUBS = [
    {"name": "Netflix", "amount": 119.9, "cycle_days": 30},
    {"name": "Spotify", "amount": 59.9, "cycle_days": 30},
]

MERCHANTS = ["market", "kafe", "kitap", "ulasim", "giyim", "kirtasiye"]

def generate_transactions(days: int = 120) -> pd.DataFrame:
    start = datetime.now() - timedelta(days=days)
    rows = []
    for day in range(days):
        d = start + timedelta(days=day)
        # günlük ufak harcamalar
        if random.random() < 0.65:
            rows.append({
                "user": USERS[0],
                "date": d.date(),
                "amount": round(random.uniform(30, 300), 2),
                "merchant": random.choice(MERCHANTS)
            })
        # abonelikler
        for s in SUBS:
            if day % s["cycle_days"] == 0:
                rows.append({
                    "user": USERS[0],
                    "date": d.date(),
                    "amount": s["amount"],
                    "merchant": s["name"]
                })
    # riskli pattern: art arda yüksek harcama
    for i in range(3):
        d = datetime.now() - timedelta(days=10 - i)
        rows.append({
            "user": USERS[0],
            "date": d.date(),
            "amount": 1500 + i * 250,
            "merchant": "elektronik"
        })
    return pd.DataFrame(rows)

if __name__ == "__main__":
    os.makedirs("data", exist_ok=True)
    df = generate_transactions(days=150)
    out = os.path.join("data", "transactions.csv")
    df.to_csv(out, index=False)
    print(f"✅ transactions.csv üretildi: {out} (kayıt: {len(df)})")
