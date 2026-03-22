# TTM Squeeze — Bollinger/Keltner coil breakout detector
# Edge: compression = energy building. Breakout direction = high-momentum move.

import pandas as pd
import numpy as np
from .base import BaseStrategy, Vote


class TTMSqueeze(BaseStrategy):
    name   = "ttm_squeeze"
    weight = 0.16

    def vote(self, df: pd.DataFrame, ob=None) -> Vote:
        import ta
        close = df["close"]

        bb   = ta.volatility.BollingerBands(close=close, window=20, window_dev=2.0)
        bb_u = bb.bollinger_hband()
        bb_l = bb.bollinger_lband()
        kc_u, kc_l = self.keltner(df, 20, 1.5)

        squeeze = (bb_u < kc_u) & (bb_l > kc_l)

        # TTM momentum oscillator
        highest = df["high"].rolling(20).max()
        lowest  = df["low"].rolling(20).min()
        mid     = (highest + lowest) / 2
        delta   = close - (mid + self.ema(close, 20)) / 2
        mom     = delta.rolling(5).mean()

        if len(mom.dropna()) < 5:
            return Vote("HOLD", 0.0)

        sq_now, sq_prev = squeeze.iloc[-1], squeeze.iloc[-2]
        m_now,  m_prev  = mom.iloc[-1],    mom.iloc[-2]
        m_prev2         = mom.iloc[-3]

        just_broke = sq_prev and not sq_now
        # Momentum histogram: is it growing?
        accel = abs(m_now) > abs(m_prev) > abs(m_prev2)

        if just_broke:
            if m_now > 0:
                conf = 0.95 if accel else 0.75
                return Vote("BUY",  conf, f"squeeze_break mom={m_now:.2f}")
            if m_now < 0:
                conf = 0.95 if accel else 0.75
                return Vote("SELL", conf, f"squeeze_break mom={m_now:.2f}")

        if sq_now:
            return Vote("HOLD", 0.0, "in_squeeze")

        # Post-breakout continuation
        if m_now > 0 and accel:
            return Vote("BUY",  0.55, f"squeeze_cont mom={m_now:.2f}")
        if m_now < 0 and accel:
            return Vote("SELL", 0.55, f"squeeze_cont mom={m_now:.2f}")

        return Vote("HOLD", 0.0)
