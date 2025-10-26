from alpaca.data import CryptoHistoricalDataClient
from alpaca.data.requests import CryptoBarsRequest
from alpaca.data.timeframe import TimeFrame
from datetime import datetime, timedelta
import pandas as pd
import os
from dotenv import load_dotenv

load_dotenv()
API_KEY = os.getenv("ALPACA_API_KEY")
SECRET_KEY = os.getenv("ALPACA_SECRET_KEY")

crypto_client = CryptoHistoricalDataClient(API_KEY, SECRET_KEY)


def get_price(ticker: str) -> float | None:
    """Fetch latest VWAP price for the given crypto symbol."""
    now = pd.Timestamp.now(tz="America/New_York")
    start = now - pd.Timedelta(minutes=15)
    request = CryptoBarsRequest(
        symbol_or_symbols=[ticker],
        timeframe=TimeFrame.Minute,
        start=start,
        end=now,
    )
    bars = crypto_client.get_crypto_bars(request).df
    if bars.empty:
        return None
    return bars["vwap"].iloc[-1]


def get_sma(ticker: str, minutes: int) -> float | None:
    """Compute simple moving average of VWAP over the last N minutes."""
    now = pd.Timestamp.now(tz="America/New_York")
    start = now - pd.Timedelta(minutes=minutes)
    request = CryptoBarsRequest(
        symbol_or_symbols=[ticker],
        timeframe=TimeFrame.Minute,
        start=start,
        end=now,
    )
    bars = crypto_client.get_crypto_bars(request).df
    if bars.empty:
        return None
    return bars["vwap"].mean()
