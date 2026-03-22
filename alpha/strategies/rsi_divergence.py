# RSI Divergence — regular + hidden divergence on price/RSI pivots
# Edge: divergence = momentum exhaustion before reversal. High win rate.

import pandas as pd
from .base import BaseStrategy, Vote


class RSIDivergence(BaseStrategy):
    name   = "rsi_div"
    weight = 0.14

    def vote(self, df: pd.DataFrame, ob=None) -> Vote:
        close = df["close"]
        rsi   = self.rsi(close, 14)

        pl_price = self.pivot_lows(close)
        ph_price = self.pivot_highs(close)
        pl_rsi   = self.pivot_lows(rsi)
        ph_rsi   = self.pivot_highs(rsi)

        pl_pi = pl_price.dropna().index
        ph_pi = ph_price.dropna().index
        pl_ri = pl_rsi.dropna().index
        ph_ri = ph_rsi.dropna().index

        signal, conf, reason = "HOLD", 0.0, ""

        if len(pl_pi) >= 2 and len(pl_ri) >= 2:
            pp1, pp2 = pl_price[pl_pi[-2]], pl_price[pl_pi[-1]]
            rp1, rp2 = pl_rsi[pl_ri[-2]],  pl_rsi[pl_ri[-1]]
            if pp2 < pp1 and rp2 > rp1:   # regular bullish
                signal, conf, reason = "BUY",  0.88, "reg_bull_div"
            elif pp2 > pp1 and rp2 < rp1: # hidden bullish
                signal, conf, reason = "BUY",  0.65, "hidden_bull_div"

        if len(ph_pi) >= 2 and len(ph_ri) >= 2:
            pp1, pp2 = ph_price[ph_pi[-2]], ph_price[ph_pi[-1]]
            rp1, rp2 = ph_rsi[ph_ri[-2]],  ph_rsi[ph_ri[-1]]
            if pp2 > pp1 and rp2 < rp1:   # regular bearish
                signal, conf, reason = "SELL", 0.88, "reg_bear_div"
            elif pp2 < pp1 and rp2 > rp1: # hidden bearish
                signal, conf, reason = "SELL", 0.65, "hidden_bear_div"

        return Vote(signal, conf, reason)
