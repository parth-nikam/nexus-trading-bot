# Microstructure — order book imbalance + bid-ask pressure
# Edge: order book tells you WHERE the market wants to go BEFORE price moves.
# This is the signal that HFT firms pay millions for. We get it free.

import pandas as pd
from .base import BaseStrategy, Vote


class Microstructure(BaseStrategy):
    name   = "microstructure"
    weight = 0.10

    # Depth levels to aggregate (top N levels of order book)
    DEPTH = 10

    def vote(self, df: pd.DataFrame, ob: dict | None = None) -> Vote:
        if ob is None:
            return Vote("HOLD", 0.0, "no_orderbook")

        bids = ob.get("bids", [])[:self.DEPTH]
        asks = ob.get("asks", [])[:self.DEPTH]

        if not bids or not asks:
            return Vote("HOLD", 0.0)

        bid_vol = sum(q for _, q in bids)
        ask_vol = sum(q for _, q in asks)
        total   = bid_vol + ask_vol

        if total == 0:
            return Vote("HOLD", 0.0)

        # Order book imbalance: > 0.6 = buy pressure, < 0.4 = sell pressure
        obi = bid_vol / total

        # Weighted mid price vs last trade
        wmid = (
            sum(p * q for p, q in bids) / bid_vol * 0.5 +
            sum(p * q for p, q in asks) / ask_vol * 0.5
        )
        last = df["close"].iloc[-1]
        spread_pct = (asks[0][0] - bids[0][0]) / last * 100

        # Only trade when spread is tight (< 0.05%)
        if spread_pct > 0.05:
            return Vote("HOLD", 0.0, f"spread={spread_pct:.3f}%")

        if obi > 0.65 and wmid > last:
            conf = min(1.0, (obi - 0.5) * 4)
            return Vote("BUY",  conf, f"obi={obi:.2f} wmid={wmid:.2f}")
        if obi < 0.35 and wmid < last:
            conf = min(1.0, (0.5 - obi) * 4)
            return Vote("SELL", conf, f"obi={obi:.2f} wmid={wmid:.2f}")

        return Vote("HOLD", 0.0, f"obi={obi:.2f}")
