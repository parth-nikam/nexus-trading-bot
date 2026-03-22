# alpha/strategies/base.py — Strategy contract + shared indicator helpers

from abc import ABC, abstractmethod
from dataclasses import dataclass
import numpy as np
import pandas as pd
import ta


@dataclass
class Vote:
    signal:   str    # "BUY" | "SELL" | "HOLD"
    conf:     float  # 0.0 – 1.0
    reason:   str = ""


class BaseStrategy(ABC):
    name:   str = "base"
    weight: float = 0.0

    @abstractmethod
    def vote(self, df: pd.DataFrame, ob: dict | None = None) -> Vote:
        """
        df  — OHLCV DataFrame, sorted ascending, 200 rows
        ob  — optional order book {"bids": [(price, qty)...], "asks": [...]}
        """
        ...

    # ── Shared helpers ────────────────────────────────────────────────────────

    @staticmethod
    def ema(s: pd.Series, n: int) -> pd.Series:
        return s.ewm(span=n, adjust=False).mean()

    @staticmethod
    def rsi(s: pd.Series, n: int = 14) -> pd.Series:
        return ta.momentum.RSIIndicator(close=s, window=n).rsi()

    @staticmethod
    def atr(df: pd.DataFrame, n: int = 14) -> pd.Series:
        return ta.volatility.AverageTrueRange(
            high=df["high"], low=df["low"], close=df["close"], window=n
        ).average_true_range()

    @staticmethod
    def adx(df: pd.DataFrame, n: int = 14) -> pd.Series:
        ind = ta.trend.ADXIndicator(
            high=df["high"], low=df["low"], close=df["close"], window=n
        )
        return ind.adx(), ind.adx_pos(), ind.adx_neg()

    @staticmethod
    def vwap(df: pd.DataFrame) -> pd.Series:
        tp  = (df["high"] + df["low"] + df["close"]) / 3
        vol = df["volume"].replace(0, np.nan).fillna(1e-9)
        return (tp * vol).cumsum() / vol.cumsum()

    @staticmethod
    def keltner(df: pd.DataFrame, n: int = 20, mult: float = 1.5):
        mid = BaseStrategy.ema(df["close"], n)
        atr = BaseStrategy.atr(df, n)
        return mid + mult * atr, mid - mult * atr

    @staticmethod
    def pivot_lows(s: pd.Series, left: int = 5, right: int = 5) -> pd.Series:
        out = pd.Series(np.nan, index=s.index)
        for i in range(left, len(s) - right):
            w = s.iloc[i - left: i + right + 1]
            if s.iloc[i] == w.min():
                out.iloc[i] = s.iloc[i]
        return out

    @staticmethod
    def pivot_highs(s: pd.Series, left: int = 5, right: int = 5) -> pd.Series:
        out = pd.Series(np.nan, index=s.index)
        for i in range(left, len(s) - right):
            w = s.iloc[i - left: i + right + 1]
            if s.iloc[i] == w.max():
                out.iloc[i] = s.iloc[i]
        return out
