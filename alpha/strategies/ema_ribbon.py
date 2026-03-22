# EMA Ribbon — 4-EMA trend stack with momentum acceleration filter
# Edge: full ribbon alignment = high-probability trend continuation.

import pandas as pd
from .base import BaseStrategy, Vote


class EMARibbon(BaseStrategy):
    name   = "ema_ribbon"
    weight = 0.18

    def vote(self, df: pd.DataFrame, ob=None) -> Vote:
        c = df["close"]
        e8, e21, e55, e89 = self.ema(c,8), self.ema(c,21), self.ema(c,55), self.ema(c,89)

        p   = c.iloc[-1]
        v8, v21, v55, v89 = e8.iloc[-1], e21.iloc[-1], e55.iloc[-1], e89.iloc[-1]

        if any(pd.isna(x) for x in [v8, v21, v55, v89]):
            return Vote("HOLD", 0.0)

        bull = sum([v8>v21, v21>v55, v55>v89, p>v8])
        bear = sum([v8<v21, v21<v55, v55<v89, p<v8])

        # Momentum: slope of 8-EMA over last 3 bars
        slope = (e8.iloc[-1] - e8.iloc[-4]) / e8.iloc[-4] * 100

        if bull == 4:
            conf = min(1.0, 0.65 + abs(slope) * 5)
            return Vote("BUY",  conf, f"ribbon=bull slope={slope:.3f}%")
        if bear == 4:
            conf = min(1.0, 0.65 + abs(slope) * 5)
            return Vote("SELL", conf, f"ribbon=bear slope={slope:.3f}%")
        if bull >= 3:
            return Vote("BUY",  0.35, "ribbon=partial-bull")
        if bear >= 3:
            return Vote("SELL", 0.35, "ribbon=partial-bear")
        return Vote("HOLD", 0.0)
