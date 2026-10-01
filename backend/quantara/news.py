from hashlib import sha256
from functools import lru_cache
import os
import re

import httpx
import numpy as np
import pandas as pd

from .market import aligned_prices
from .model_runtime import serialized_training
from .schemas import NewsItem

DISTILBERT_MODEL = "navu013/finbert-sentiment-distilbert"


@lru_cache(maxsize=1)
@serialized_training
def local_classifier():
    from transformers import AutoTokenizer, pipeline

    tokenizer = AutoTokenizer.from_pretrained(DISTILBERT_MODEL)
    # This checkpoint ships a BERT tokenizer; DistilBERT has no segment-ID input.
    tokenizer.model_input_names = ["input_ids", "attention_mask"]
    return pipeline(
        "text-classification", model=DISTILBERT_MODEL, tokenizer=tokenizer, device=-1
    )


def classify(text, method="baseline"):
    if method == "distilbert":
        result = local_classifier()(text, truncation=True)[0]
        label = result["label"].lower()
        sign = 1 if label == "positive" else -1 if label == "negative" else 0
        return {
            "score": sign * float(result["score"]),
            "label": label,
            "model": DISTILBERT_MODEL,
        }
    tokens = re.findall(r"[a-z]+", text.lower())
    positive = {
        "growth",
        "beat",
        "profit",
        "upgrade",
        "strong",
        "gains",
        "bullish",
        "surge",
    }
    negative = {
        "loss",
        "miss",
        "decline",
        "downgrade",
        "weak",
        "fraud",
        "bearish",
        "crash",
    }
    score = sum(
        1 if t in positive else -1 if t in negative else 0 for t in tokens
    ) / max(1, sum(t in positive | negative for t in tokens))
    return {
        "score": score,
        "label": "positive" if score > 0 else "negative" if score < 0 else "neutral",
        "model": "transparent-lexicon-v1",
    }


def ingest(items, method, store, owner):
    result = []
    for item in items:
        digest = sha256(
            (
                " ".join(item.text.lower().split())
                + "|"
                + ",".join(sorted(item.symbols))
                + "|"
                + item.timestamp.date().isoformat()
            ).encode()
        ).hexdigest()
        identifier = sha256((owner + ":" + digest).encode()).hexdigest()
        try:
            old = store.get("news", identifier, owner)
            result.append(old)
            continue
        except KeyError:
            pass
        value = (
            item.model_dump(mode="json")
            | classify(item.text, method)
            | {"content_hash": digest}
        )
        result.append(store.create("news", owner, value, identifier))
    return result


def impact(dataset, news, horizon):
    prices = aligned_prices(dataset)
    x, y, stamps = [], [], []
    for item in sorted(news, key=lambda i: i["available_at"]):
        at = pd.Timestamp(item["available_at"])
        start = prices.index.searchsorted(at)
        if start + horizon >= len(prices):
            continue
        for symbol in item["symbols"]:
            if symbol in prices:
                x.append([1, item["score"]])
                y.append(
                    float(
                        prices[symbol].iloc[start + horizon]
                        / prices[symbol].iloc[start]
                        - 1
                    )
                )
                stamps.append((prices.index[start], prices.index[start + horizon]))
    if len(x) < 15:
        raise ValueError(
            "At least 15 timestamped, price-aligned news examples are required"
        )
    order = np.argsort([a.value for a, _ in stamps])
    x, y = np.array(x)[order], np.array(y)[order]
    stamps = [stamps[i] for i in order]
    cut = int(len(y) * 0.8)
    boundary = stamps[cut][0]
    # Purge training labels whose outcome is not known before the test starts.
    train = np.array([b < boundary for _, b in stamps[:cut]])
    tx, ty = x[:cut][train], y[:cut][train]
    if len(ty) < 5:
        raise ValueError("Insufficient non-overlapping training labels")
    coef = np.linalg.solve(tx.T @ tx + np.diag([1e-9, 1e-2]), tx.T @ ty)
    pred = x[cut:] @ coef
    return {
        "method": "ridge_sentiment_v1",
        "horizon_bars": horizon,
        "train_rows": len(ty),
        "test_rows": len(y) - cut,
        "test_start": boundary.isoformat(),
        "mae": float(np.abs(pred - y[cut:]).mean()),
        "baseline_mae": float(np.abs(ty.mean() - y[cut:]).mean()),
        "coefficients": coef.tolist(),
        "scope": "Association evaluation; no claim of causation or outperformance",
        "label_definition": "Close-to-close total return from the first bar starting at/after news availability to horizon_bars later; overlapping labels purged before the test boundary",
    }


def alpaca_news(symbols, start, end):
    key, secret = os.getenv("ALPACA_KEY"), os.getenv("ALPACA_SECRET")
    if not key or not secret:
        raise ValueError("Alpaca news credentials are missing")
    response = httpx.get(
        "https://data.alpaca.markets/v1beta1/news",
        params={"symbols": ",".join(symbols), "start": start, "end": end, "limit": 50},
        headers={"APCA-API-KEY-ID": key, "APCA-API-SECRET-KEY": secret},
        timeout=30,
    )
    response.raise_for_status()
    return [
        NewsItem(
            external_id=str(n["id"]),
            timestamp=n["created_at"],
            available_at=n["updated_at"],
            symbols=[s for s in n["symbols"] if s in symbols],
            text=n["headline"] + " " + n.get("summary", ""),
            source="alpaca-news",
            url=n.get("url", ""),
        )
        for n in response.json()["news"]
        if set(n["symbols"]) & set(symbols)
    ]
