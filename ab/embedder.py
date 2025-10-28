# embedder.py
"""
Gemini embedding çağrılarını toplu (batch) ve güvenli (retry/backoff) şekilde saran yardımcılar.
google-generativeai -> gemini_client.embed_texts fonksiyonunu kullanır.
"""

from __future__ import annotations
import time
from typing import List, Sequence, Iterable, Callable, Optional

# Tekil embedding çağrımızı buradan içe aktarıyoruz
from gemini_client import embed_texts as _embed_texts


class Embedder:
    def __init__(
        self,
        batch_size: int = 32,
        max_retries: int = 3,
        base_sleep: float = 0.5,
        per_call_sleep: float = 0.02,
        embed_fn: Optional[Callable[[Sequence[str], float], List[List[float]]]] = None,
    ) -> None:
        """
        :param batch_size: Her partide kaç metin işlenecek.
        :param max_retries: Hata durumunda yeniden deneme sayısı.
        :param base_sleep: Retry backoff için başlangıç bekleme süresi (sn).
        :param per_call_sleep: Başarılı her alt çağrıdan sonra nazik bekleme (sn).
        :param embed_fn: Dışarıdan farklı bir embedding fonksiyonu enjekte etmek için (opsiyonel).
        """
        self.batch_size = batch_size
        self.max_retries = max_retries
        self.base_sleep = base_sleep
        self.per_call_sleep = per_call_sleep
        self._embed_fn = embed_fn or _embed_texts

    def _embed_once(self, texts: Sequence[str]) -> List[List[float]]:
        # gemini_client.embed_texts(texts, sleep=per_call_sleep)
        return self._embed_fn(texts, sleep=self.per_call_sleep)

    def embed_batched(self, texts: Sequence[str]) -> List[List[float]]:
        """
        Büyük bir metin listesini parçalara bölerek embedding üretir.
        Hata durumunda üssel (exponential) backoff ile yeniden dener.
        """
        vectors: List[List[float]] = []
        if not texts:
            return vectors

        for i in range(0, len(texts), self.batch_size):
            chunk = texts[i : i + self.batch_size]
            attempt = 0
            while True:
                try:
                    vecs = self._embed_once(chunk)
                    vectors.extend(vecs)
                    break
                except Exception as e:
                    attempt += 1
                    if attempt > self.max_retries:
                        raise RuntimeError(f"Embedding başarısız (denemeler bitti): {e}") from e
                    # üssel backoff
                    sleep_sec = self.base_sleep * (2 ** (attempt - 1))
                    time.sleep(sleep_sec)
        return vectors


# --------- Hızlı test ---------
if __name__ == "__main__":
    demo_texts = [
        "Ekstre nedir?",
        "Kredi karti limitim dolarsa ne olur?",
        "Kazanilan Harçlik (bonus) nasil kullanilir?"
    ]
    emb = Embedder(batch_size=2, max_retries=3, base_sleep=0.5, per_call_sleep=0.02)
    vecs = emb.embed_batched(demo_texts)
    dims = len(vecs[0]) if vecs else 0
    print(f"✅ embedder.py OK — {len(vecs)} adet vektör üretildi, boyut: {dims}")
