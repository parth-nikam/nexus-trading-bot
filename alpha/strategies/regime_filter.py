# Regime Filter — ADX-gated trend/range mode switcher
# Edge: different strategies work in different regimes. This routes correctly.

import pandas as pd
import ta
from .base import BaseStrategy, Vote


class RegimeFilter(BaseStrategy):
    name   = "regime"
    weight = 0.14

    def vote(self, df: pd.DataFrame, ob=None) -> Vote:
        close = df["close"]
        adx_v, dip, din = self.adx(df, 14)
        rsi   = self.rsi(close, 14)
        bb    = ta.volatility.BollingerBands(close=close, window=20, window_dev=2.0)
        e20   = self.ema(close, 20)
        e50   = self.ema(close, 50)

        a   = adx_v.iloc[-1]
        r   = rsi.iloc[-1]
        p   = close.iloc[-1]
        bbu = bb.bollinger_hband().iloc[-1]
        bbl = bb.bollinger_lband().iloc[-1]
        d20 = e20.iloc[-1]
        d50 = e50.iloc[-1]
        dp  = dip.iloc[-1]
        dn  = din.iloc[-1]

        if pd.isna(a):
            return Vote("HOLD", 0.0)

        if a > 25:
            # Strong trend — follow DI crossover
            if dp > dn and d20 > d50 and p > d20:
                return Vote("BUY",  min(1.0, a / 50), f"trend adx={a:.0f} +DI>{dn:.0f}")
            if dn > dp and d20 < d50 and p < d20:
                return Vote("SELL", min(1.0, a / 50), f"trend adx={a:.0f} -DI>{dp:.0f}")
            return Vote("HOLD", 0.0)

        if a < 20:
            # Range — fade extremes
            if p <= bbl * 1.002 and r < 35:
                return Vote("BUY",  0.72, f"range bbl={bbl:.0f} rsi={r:.0f}")
            if p >= bbu * 0.998 and r > 65:
                return Vote("SELL", 0.72, f"range bbu={bbu:.0f} rsi={r:.0f}")
            return Vote("HOLD", 0.0)

        return Vote("HOLD", 0.0, f"transitioning adx={a:.0f}")
