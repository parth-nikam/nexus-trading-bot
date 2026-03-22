# alpha/engine.py — Strategy orchestrator with weighted voting + Kelly sizing
#
# Architecture:
#   1. Each strategy votes independently (no shared state between strategies)
#   2. Votes are weighted and aggregated into a net score
#   3. Score must clear threshold AND pass regime sanity check
#   4. Kelly criterion sizes the position based on historical win rate

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional
import pandas as pd

import config
from alpha.strategies import (
    VWAPDeviation, EMARibbon, TTMSqueeze,
    RSIDivergence, RegimeFilter, Microstructure, FundingRate,
)
from alpha.strategies.base import Vote
from utils.logger import get_logger

logger = get_logger(__name__)

BUY  = "BUY"
SELL = "SELL"
HOLD = "HOLD"


@dataclass
class Signal:
    symbol:     str
    action:     str          # BUY | SELL | HOLD
    score:      float        # 0.0 – 1.0 weighted confidence
    kelly_f:    float        # Kelly fraction for position sizing
    breakdown:  dict         # per-strategy votes
    reason:     str = ""


@dataclass
class StrategyStats:
    """Rolling win rate tracker per strategy — feeds Kelly calculator."""
    wins:   int = 0
    losses: int = 0

    @property
    def win_rate(self) -> float:
        total = self.wins + self.losses
        return self.wins / total if total > 0 else 0.5

    @property
    def kelly_fraction(self) -> float:
        """
        Kelly criterion: f* = W - (1-W)/R
        W = win rate, R = win/loss ratio (assume 2:1 for crypto)
        Halved (half-Kelly) for safety.
        """
        w = self.win_rate
        r = 2.0  # assume 2:1 reward/risk
        f = w - (1 - w) / r
        return max(0.0, min(0.25, f / 2))  # half-Kelly, capped at 25%


class AlphaEngine:
    """
    Runs all strategies on each tick, aggregates votes, returns Signal.
    Thread-safe: each call is stateless (strategies don't mutate df).
    """

    MIN_CANDLES = 100

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
        ]
        # Per-strategy rolling stats for Kelly sizing
        self._stats: dict[str, StrategyStats] = {
            s.name: StrategyStats() for s in self._strategies
        }
        # Verify weights sum to ~1.0
        total_w = sum(s.weight for s in self._strategies)
        assert abs(total_w - 1.0) < 0.01, f"Weights sum to {total_w}, must be 1.0"

    def update_funding(self, rate: float):
        self._funding.update_rate(rate)

    def record_outcome(self, strategy_name: str, won: bool):
        """Call after a trade closes to update Kelly stats."""
        if strategy_name in self._stats:
            if won:
                self._stats[strategy_name].wins += 1
            else:
                self._stats[strategy_name].losses += 1

    def evaluate(
        self,
        symbol: str,
        df: pd.DataFrame,
        ob: dict | None = None,
    ) -> Signal:
        if len(df) < self.MIN_CANDLES:
            logger.warning(f"[{symbol}] Only {len(df)} candles, need {self.MIN_CANDLES}")
            return Signal(symbol, HOLD, 0.0, 0.0, {})

        votes: dict[str, Vote] = {}
        for strat in self._strategies:
            try:
                votes[strat.name] = strat.vote(df, ob)
            except Exception as e:
                logger.error(f"[{symbol}] Strategy {strat.name} error: {e}")
                votes[strat.name] = Vote(HOLD, 0.0, f"error: {e}")

        signal, score, breakdown = self._aggregate(votes)

        # Kelly fraction = average of contributing strategies' Kelly fractions
        kelly = self._kelly_for_signal(signal, votes)

        reason = " | ".join(
            f"{k}={v.signal}({v.conf:.2f})"
            for k, v in votes.items()
            if v.signal != HOLD
        )

        logger.info(
            f"[{symbol}] {signal} score={score:.3f} kelly={kelly:.3f} | {reason or 'all_hold'}"
        )

        return Signal(symbol, signal, score, kelly, breakdown, reason)

    def _aggregate(self, votes: dict[str, Vote]) -> tuple[str, float, dict]:
        buy_score  = 0.0
        sell_score = 0.0
        breakdown  = {}

        for strat in self._strategies:
            v = votes.get(strat.name, Vote(HOLD, 0.0))
            breakdown[strat.name] = {"signal": v.signal, "conf": v.conf, "reason": v.reason}
            if v.signal == BUY:
                buy_score  += strat.weight * v.conf
            elif v.signal == SELL:
                sell_score += strat.weight * v.conf

        if buy_score >= config.BUY_THRESHOLD:
            return BUY,  buy_score,  breakdown
        if sell_score >= config.SELL_THRESHOLD:
            return SELL, sell_score, breakdown
        return HOLD, max(buy_score, sell_score), breakdown

    def _kelly_for_signal(self, signal: str, votes: dict[str, Vote]) -> float:
        if signal == HOLD:
            return 0.0
        fractions = [
            self._stats[s.name].kelly_fraction
            for s in self._strategies
            if votes.get(s.name, Vote(HOLD, 0.0)).signal == signal
        ]
        if not fractions:
            return config.TRADE_SIZE_PCT
        return max(config.TRADE_SIZE_PCT * 0.5, min(config.TRADE_SIZE_PCT, sum(fractions) / len(fractions)))
