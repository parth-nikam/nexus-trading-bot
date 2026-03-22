# VWAP Deviation — mean reversion against volume-weighted price anchor
# Edge: institutional algos defend VWAP. Deviations > 2 ATR snap back.

import pandas as pd
from .base import BaseStrategy, Vote


class VWAPDeviation(BaseStrategy):
    name   = "vwap_dev"
    weight = 0.18

    def vote(self, df: pd.DataFrame, ob=None) -> Vote:
        close = df["close"]
        vwap  = self.vwap(df)
        atr   = self.atr(df, 14)
        rsi   = self.rsi(close, 14)

        p, v, a, r = close.iloc[-1], vwap.iloc[-1], atr.iloc[-1], rsi.iloc[-1]
        if a == 0 or pd.isna(v):
            return Vote("HOLD", 0.0)

        dev = (v - p) / a  # positive = price below VWAP

        if dev > 2.0 and r < 40:
            return Vote("BUY",  min(1.0, dev / 4), f"dev={dev:.2f}σ RSI={r:.0f}")
        if dev > 1.5 and r < 45:
            return Vote("BUY",  min(0.7, dev / 4), f"dev={dev:.2f}σ RSI={r:.0f}")
        if dev < -1.5 or r > 62:
            return Vote("SELL", min(1.0, abs(dev) / 3), f"dev={dev:.2f}σ RSI={r:.0f}")
        return Vote("HOLD", 0.0)
