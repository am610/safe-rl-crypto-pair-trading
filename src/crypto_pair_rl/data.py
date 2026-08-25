"""Public cryptocurrency market data access."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from urllib.parse import urlencode
from urllib.request import Request, urlopen

import pandas as pd


COINBASE_CANDLES = "https://api.exchange.coinbase.com/products/{product}/candles"


def fetch_coinbase_candles(product: str, granularity: int = 3600) -> pd.DataFrame:
    """Download the latest Coinbase candles without credentials."""
    if granularity not in {60, 300, 900, 3600, 21600, 86400}:
        raise ValueError("Unsupported Coinbase granularity")
    query = urlencode({"granularity": granularity})
    request = Request(
        f"{COINBASE_CANDLES.format(product=product)}?{query}",
        headers={"User-Agent": "crypto-multi-pair-research/0.1"},
    )
    with urlopen(request, timeout=30) as response:
        payload = json.load(response)
    if not isinstance(payload, list):
        raise RuntimeError(f"Coinbase returned an unexpected response: {payload}")
    frame = pd.DataFrame(payload, columns=["time", "low", "high", "open", "close", "volume"])
    frame["time"] = pd.to_datetime(frame["time"], unit="s", utc=True)
    frame = frame.set_index("time").sort_index()
    frame.index.name = "timestamp_utc"
    for column in ["low", "high", "open", "close", "volume"]:
        frame[column] = pd.to_numeric(frame[column], errors="raise")
    frame.attrs["product"] = product
    frame.attrs["retrieved_at_utc"] = datetime.now(timezone.utc).isoformat()
    frame.attrs["provider"] = "Coinbase Exchange public API"
    return frame
