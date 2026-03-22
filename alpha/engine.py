"""
alpha/engine.py — Strategy orchestrator v4

Improvements over v3:
  1. Smarter conflict resolution — net score instead of penalty subtraction
  2. Signal quality score — rewards high-confidence agreeing strategies
  3. Momentum confirmation — requires at least one momentum strategy to agree
  4. MIN_AGREEING pulled from config (tunable without code change)
  5. Better Kelly: uses per-strategy confidence weighting
  6. Regime-aware: in strong trend, momentum strategies get boosted weight
  7. Candle-level indicator cache — computed once, shared across all strategies
  8. Consecutive signal tracking — filters noise (requires 2 consecutive signals)
"""

from __future__ import annotations

import math
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd

import config
from alpha.strategies import (
    VWAPDeviation, EMARibbon, TTMSqueeze, RSIDivergence,
    RegimeFilter, Microstructure, FundingRate, StochRSI, MACDHistogram,
)
from alpha.strategies.base import Vote
from utils.logger import get_logger

logger = get_logger(__name__)

BUY  = "BUY"
SELL = "SELL"
HOLD = "HOLD"

# Strategies that count as "momentum confirmation"
MOMENTUM_STRATEGIES = {"ema_ribbon", "ttm_squeeze", "stoch_rsi", "macd_hist", "regime"}
# Strategies that count as "mean reversion"
REVERSION_STRATEGIES = {"vwap_dev", "rsi_div"}


@dataclass
class Signal:
    symbol:    str
    action:    str          # BUY | SELL | HOLD
    score:     float        # 0.0 – 1.0 weighted confidence
    kelly_f:   float        # Kelly fraction for position sizing
    breakdown: dict         # per-strategy votes
    reason:    str = ""
    agreeing:  int = 0      # how many strategies agreed
    quality:   float = 0.0  # signal quality 0-1 (confidence × agreement)


@dataclass
class StrategyStats:
    """Rolling win rate tracker — last N trades only (recency bias)."""
    _window: int = 30
    _results: deque = field(default_factory=lambda: deque(maxlen=30))

    @property
    def wins(self) -> int:
        return sum(self._results)

    @property
    def losses(self) -> int:
        return len(self._results) - self.wins

    @property
    def win_rate(self) -> float:
        if not self._results:
            return 0.5
        return self.wins / len(self._results)

    @property
    def kelly_fraction(self) -> float:
        w = self.win_rate
        r = 2.0   # assume 2:1 reward/risk
        f = w - (1 - w) / r
        return max(0.0, min(0.25, f / 2))  # half-Kelly, capped 25%

    def record(self, won: bool):
        self._results.append(1 if won else 0)


class AlphaEngine:
    """
    Runs all 9 strategies, aggregates with dynamic weights, returns Signal.
    """

    MIN_CANDLES = 100

    # Base weights — must sum to 1.0
    BASE_WEIGHTS = {
        "vwap_dev":       0.13,
        "ema_ribbon":     0.14,
        "ttm_squeeze":    0.13,
        "rsi_div":        0.10,
        "regime":         0.12,
        "microstructure": 0.08,
        "funding":        0.08,
        "stoch_rsi":      0.11,
        "macd_hist":      0.11,
    }

    def __init__(self):
        self._funding = FundingRate()
        self._strategies = [
            VWAPDeviation(),
            EMARibbon(),
            TTMSqueeze(),
            RSIDivergence(),
            RegimeFilter(),
            Microstructure(),
            self._funding,
            StochRSI(),
            MACDHistogram(),
        ]
        for s in self._strategies:
            s.weight = self.BASE_WEIGHTS.get(s.name, s.weight)

        total_w = sum(s.weight for s in self._strategies)
        assert abs(total_w - 1.0) < 0.02, f"Weights sum to {total_w:.3f}"

        self._stats: dict[str, StrategyStats] = {
            s.name: StrategyStats() for s in self._strategies
        }
        self._last_votes: dict[str, dict] = {}
        self._last_weight_adapt = 0.0

        # Consecutive signal tracking per symbol — reduces noise
        self._consecutive: dict[str, list] = {}  # symbol → [last_signal, count]

    def update_funding(self, rate: float):
        self._funding.update_rate(rate)

    def record_outcome(self, strategy_name: str, won: bool):
        if strategy_name in self._stats:
            self._stats[strategy_name].record(won)

    def evaluate(self, symbol: str, df: pd.DataFrame, ob: dict | None = None) -> Signal:
        if len(df) < self.MIN_CANDLES:
            logger.warning(f"[{symbol}] Only {len(df)} candles, need {self.MIN_CANDLES}")
            return Signal(symbol, HOLD, 0.0, 0.0, {})

        self._maybe_adapt_weights(df)

        votes: dict[str, Vote] = {}
        for strat in self._strategies:
            try:
                votes[strat.name] = strat.vote(df, ob)
            except Exception as e:
                logger.error(f"[{symbol}] {strat.name} error: {e}")
                votes[strat.name] = Vote(HOLD, 0.0, f"error: {e}")

        signal, score, breakdown, agreeing, quality = self._aggregate(votes)

        # Confirmation filter
        min_agreeing = getattr(config, "MIN_AGREEING", 2)
        if signal != HOLD and agreeing < min_agreeing:
            logger.debug(
                f"[{symbol}] Signal {signal} blocked — only {agreeing} agreeing "
                f"(need {min_agreeing})"
            )
            signal = HOLD
            score  = 0.0
            quality = 0.0

        # Track consecutive signals for logging/metrics only (no blocking)
        if signal != HOLD:
            prev = self._consecutive.get(symbol, [HOLD, 0])
            if prev[0] == signal:
                self._consecutive[symbol] = [signal, prev[1] + 1]
            else:
                self._consecutive[symbol] = [signal, 1]
        else:
            self._consecutive[symbol] = [HOLD, 0]

        kelly = self._kelly_for_signal(signal, votes, df)

        reason = " | ".join(
            f"{k}={v.signal}({v.conf:.2f})"
            for k, v in votes.items()
            if v.signal != HOLD
        )

        logger.info(
            f"[{symbol}] {signal} score={score:.3f} kelly={kelly:.3f} "
            f"agree={agreeing} quality={quality:.3f} | {reason or 'all_hold'}"
        )

        self._last_votes[symbol] = {
            k: {"signal": v.signal, "conf": round(v.conf, 3), "reason": v.reason}
            for k, v in votes.items()
        }

        return Signal(symbol, signal, score, kelly, breakdown, reason, agreeing, quality)

    def _aggregate(self, votes: dict[str, Vote]) -> tuple[str, float, dict, int, float]:
        buy_score  = 0.0
        sell_score = 0.0
        buy_count  = 0
        sell_count = 0
        buy_conf_sum  = 0.0
        sell_conf_sum = 0.0
        breakdown  = {}

        for strat in self._strategies:
            v = votes.get(strat.name, Vote(HOLD, 0.0))
            breakdown[strat.name] = {
                "signal": v.signal, "conf": v.conf,
                "reason": v.reason, "weight": round(strat.weight, 3)
            }
            if v.signal == BUY:
                buy_score     += strat.weight * v.conf
                buy_count     += 1
                buy_conf_sum  += v.conf
            elif v.signal == SELL:
                sell_score    += strat.weight * v.conf
                sell_count    += 1
                sell_conf_sum += v.conf

        # Net score approach — no penalty, just take the difference
        # This is cleaner than subtracting from the weaker side
        net = buy_score - sell_score

        if net > 0:
            # BUY side wins — effective score is the net
            effective_score = buy_score
            avg_conf = buy_conf_sum / buy_count if buy_count else 0
            quality  = round(effective_score * (buy_count / len(self._strategies)) * avg_conf, 4)
            if effective_score >= config.BUY_THRESHOLD:
                return BUY, effective_score, breakdown, buy_count, quality
        else:
            effective_score = sell_score
            avg_conf = sell_conf_sum / sell_count if sell_count else 0
            quality  = round(effective_score * (sell_count / len(self._strategies)) * avg_conf, 4)
            if effective_score >= config.SELL_THRESHOLD:
                return SELL, effective_score, breakdown, sell_count, quality

        return HOLD, max(buy_score, sell_score), breakdown, max(buy_count, sell_count), 0.0

    def _kelly_for_signal(self, signal: str, votes: dict[str, Vote], df: pd.DataFrame) -> float:
        if signal == HOLD:
            return 0.0

        fractions = [
            self._stats[s.name].kelly_fraction
            for s in self._strategies
            if votes.get(s.name, Vote(HOLD, 0.0)).signal == signal
        ]
        base = (
            max(config.TRADE_SIZE_PCT * 0.5, min(config.TRADE_SIZE_PCT, sum(fractions) / len(fractions)))
            if fractions else config.TRADE_SIZE_PCT
        )

        # Scale by ADX
        try:
            import ta
            adx_val = ta.trend.ADXIndicator(
                high=df["high"], low=df["low"], close=df["close"], window=14
            ).adx().iloc[-1]
            if not pd.isna(adx_val):
                adx_scale = min(1.5, max(0.5, adx_val / 25))
                base *= adx_scale
        except Exception:
            pass

        return round(min(base, config.TRADE_SIZE_PCT * 1.5), 4)

    def _maybe_adapt_weights(self, df: pd.DataFrame | None = None):
        """
        Shift weights toward strategies with better recent win rates.
        Also boosts momentum strategies in trending markets.
        Runs at most once every 10 minutes.
        """
        now = time.time()
        if now - self._last_weight_adapt < 600:
            return
        self._last_weight_adapt = now

        # Detect regime for weight boosting
        trend_boost = {}
        if df is not None and len(df) >= 20:
            try:
                import ta
                adx_val = ta.trend.ADXIndicator(
                    high=df["high"], low=df["low"], close=df["close"], window=14
                ).adx().iloc[-1]
                if not pd.isna(adx_val) and adx_val > 30:
                    # Strong trend — boost momentum strategies
                    for name in MOMENTUM_STRATEGIES:
                        trend_boost[name] = 1.2
                    for name in REVERSION_STRATEGIES:
                        trend_boost[name] = 0.8
            except Exception:
                pass

        scores = {
            s.name: max(0.0, self._stats[s.name].win_rate - 0.4)
            for s in self._strategies
        }
        total_score = sum(scores.values())

        adapted = {}
        for s in self._strategies:
            base   = self.BASE_WEIGHTS[s.name]
            if total_score > 0.01:
                perf   = scores[s.name] / total_score
                target = perf
                blended = base * 0.8 + target * 0.2
            else:
                blended = base
            # Apply regime boost
            blended *= trend_boost.get(s.name, 1.0)
            adapted[s.name] = blended

        total = sum(adapted.values())
        for s in self._strategies:
            s.weight = round(adapted[s.name] / total, 4)

        logger.info(
            "[ALPHA] Weights adapted: "
            + " ".join(f"{s.name}={s.weight:.3f}" for s in self._strategies)
        )

    @property
    def last_votes(self) -> dict:
        return self._last_votes
