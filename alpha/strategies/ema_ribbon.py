# EMA Ribbon — 4-EMA trend stack with momentum acceleration filter
# Edge: full ribbon alignment = high-probability trend continuation.
# v2: partial alignment signals, slope-based confidence, volume confirmation

import pandas as pd
from .base import BaseStrategy, Vote


class EMARibbon(BaseStrategy):
    name   = "ema_ribbon"
    weight = 0.14

    def vote(self, df: pd.DataFrame, ob=None) -> Vote:
        c = df["close"]
        e8, e21, e55, e89 = self.ema(c, 8), self.ema(c, 21), self.ema(c, 55), self.ema(c, 89)

        p   = c.iloc[-1]
        v8, v21, v55, v89 = e8.iloc[-1], e21.iloc[-1], e55.iloc[-1], e89.iloc[-1]

        if any(pd.isna(x) for x in [v8, v21, v55, v89]):
            return Vote("HOLD", 0.0)

        bull = sum([v8 > v21, v21 > v55, v55 > v89, p > v8])
        bear = sum([v8 < v21, v21 < v55, v55 < v89, p < v8])

        # Slope of 8-EMA over last 3 bars (momentum)
        slope = (e8.iloc[-1] - e8.iloc[-4]) / e8.iloc[-4] * 100

        # Volume confirmation — above 20-bar average
        vol_avg = df["volume"].rolling(20).mean().iloc[-1]
        vol_now = df["volume"].iloc[-1]
        vol_ok  = vol_now > vol_avg * 0.8  # at least 80% of average

        if bull == 4:
            conf = min(0.85, 0.60 + abs(slope) * 3)
            if vol_ok:
                conf = min(0.85, conf + 0.05)
            return Vote("BUY",  conf, f"ribbon=bull slope={slope:.3f}%")
        if bear == 4:
            conf = min(0.85, 0.60 + abs(slope) * 3)
            if vol_ok:
                conf = min(0.85, conf + 0.05)
            return Vote("SELL", conf, f"ribbon=bear slope={slope:.3f}%")

        # Partial alignment with strong slope
        if bull >= 3 and slope > 0.02:
            return Vote("BUY",  0.45, f"ribbon=partial-bull slope={slope:.3f}%")
        if bear >= 3 and slope < -0.02:
            return Vote("SELL", 0.45, f"ribbon=partial-bear slope={slope:.3f}%")

        # EMA crossover signal (8 crosses 21)
        e8_prev, e21_prev = e8.iloc[-2], e21.iloc[-2]
        if e8_prev < e21_prev and v8 > v21:
            return Vote("BUY",  0.55, f"ema_cross_up 8x21")
        if e8_prev > e21_prev and v8 < v21:
            return Vote("SELL", 0.55, f"ema_cross_dn 8x21")

        return Vote("HOLD", 0.0, f"bull={bull} bear={bear}")
