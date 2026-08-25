"""Public cryptocurrency market data access."""

from __future__ import annotations

import json
import time
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


def fetch_coinbase_history(
    product: str,
    start: str | pd.Timestamp,
    end: str | pd.Timestamp,
    granularity: int = 3600,
    pause_seconds: float = 0.12,
) -> pd.DataFrame:
    """Download a longer history in Coinbase compliant time windows."""
    start_time = pd.Timestamp(start, tz="UTC") if pd.Timestamp(start).tzinfo is None else pd.Timestamp(start).tz_convert("UTC")
    end_time = pd.Timestamp(end, tz="UTC") if pd.Timestamp(end).tzinfo is None else pd.Timestamp(end).tz_convert("UTC")
    if end_time <= start_time:
        raise ValueError("End must follow start")
    window = pd.Timedelta(seconds=granularity * 299)
    frames = []
    cursor = start_time
    while cursor < end_time:
        window_end = min(cursor + window, end_time)
        query = urlencode(
            {
                "granularity": granularity,
                "start": cursor.isoformat().replace("+00:00", "Z"),
                "end": window_end.isoformat().replace("+00:00", "Z"),
            }
        )
        request = Request(
            f"{COINBASE_CANDLES.format(product=product)}?{query}",
            headers={"User-Agent": "crypto-multi-pair-research/0.1"},
        )
        with urlopen(request, timeout=30) as response:
            payload = json.load(response)
        if not isinstance(payload, list):
            raise RuntimeError(f"Coinbase returned an unexpected response: {payload}")
        frames.append(pd.DataFrame(payload, columns=["time", "low", "high", "open", "close", "volume"]))
        cursor = window_end + pd.Timedelta(seconds=granularity)
        if pause_seconds:
            time.sleep(pause_seconds)
    frame = pd.concat(frames, ignore_index=True).drop_duplicates("time")
    frame["time"] = pd.to_datetime(frame["time"], unit="s", utc=True)
    frame = frame.set_index("time").sort_index()
    frame.index.name = "timestamp_utc"
    for column in ["low", "high", "open", "close", "volume"]:
        frame[column] = pd.to_numeric(frame[column], errors="raise")
    frame.attrs.update(
        product=product,
        provider="Coinbase Exchange public API",
        retrieved_at_utc=datetime.now(timezone.utc).isoformat(),
    )
    return frame
