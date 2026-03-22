# Funding Rate — perpetual futures carry signal
# Edge: extreme funding = crowded trade. Fade it. Neutral funding = trend safe.
# Positive funding = longs pay shorts → market is overleveraged long → fade.
# Negative funding = shorts pay longs → market is overleveraged short → fade.

import pandas as pd
from .base import BaseStrategy, Vote


class FundingRate(BaseStrategy):
    name   = "funding"
    weight = 0.10

    # Annualized funding thresholds (8h rate × 3 × 365)
    EXTREME_LONG  =  0.50   # > 50% annualized = crowded long → SELL
    EXTREME_SHORT = -0.30   # < -30% annualized = crowded short → BUY
    NEUTRAL_BAND  =  0.10   # within ±10% = safe to follow trend

    def __init__(self, funding_rate: float = 0.0):
        self._rate = funding_rate  # updated externally each cycle

    def update_rate(self, rate: float):
        self._rate = rate

    def vote(self, df: pd.DataFrame, ob=None) -> Vote:
        r = self._rate
        # Annualize: 8h rate × 3 periods/day × 365 days
        annualized = r * 3 * 365

        if annualized > self.EXTREME_LONG:
            conf = min(1.0, (annualized - self.EXTREME_LONG) * 2)
            return Vote("SELL", conf, f"funding={annualized:.1%} crowded_long")

        if annualized < self.EXTREME_SHORT:
            conf = min(1.0, abs(annualized - self.EXTREME_SHORT) * 2)
            return Vote("BUY",  conf, f"funding={annualized:.1%} crowded_short")

        # Neutral funding — don't fight the trend
        return Vote("HOLD", 0.0, f"funding={annualized:.1%} neutral")
