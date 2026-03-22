"""
execution/risk.py — Portfolio-level risk manager.

Enforces:
  - Max concurrent positions
  - Daily loss limit
  - Max drawdown halt
  - Per-trade stop loss / take profit
  - Volatility-adjusted position sizing
  - Trade attribution (which strategy generated each trade)
  - Per-strategy performance metrics
"""

from __future__ import annotations

import asyncio
import math
import time
from dataclasses import dataclass, field
from typing import Optional

import config
from utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class TradeRecord:
    symbol:        str
    side:          str
    entry_price:   float
    quantity:      float       # leveraged quantity (notional / price)
    margin:        float       # actual capital locked (notional / leverage)
    stop_loss:     float
    take_profit:   float
    leverage:      int   = 1
    strategy:      str   = "unknown"
    entry_time:    float = field(default_factory=time.time)
    exit_price:    float = 0.0
    pnl:           float = 0.0
    hold_seconds:  float = 0.0
    open:          bool  = True
    trail_active:  bool  = False
    trail_stop:    float = 0.0
    peak_price:    float = 0.0


@dataclass
class StrategyMetrics:
    """Rolling performance metrics per strategy."""
    wins:       int   = 0
    losses:     int   = 0
    total_pnl:  float = 0.0
    pnl_sq_sum: float = 0.0   # for Sharpe calculation

    @property
    def trade_count(self) -> int:
        return self.wins + self.losses

    @property
    def win_rate(self) -> float:
        return self.wins / self.trade_count if self.trade_count else 0.5

    @property
    def avg_pnl(self) -> float:
        return self.total_pnl / self.trade_count if self.trade_count else 0.0

    @property
    def sharpe(self) -> float:
        """Simplified Sharpe: mean_pnl / std_pnl."""
        n = self.trade_count
        if n < 2:
            return 0.0
        mean = self.total_pnl / n
        variance = (self.pnl_sq_sum / n) - (mean ** 2)
        std = math.sqrt(max(variance, 1e-12))
        return mean / std

    def record(self, pnl: float):
        if pnl > 0:
            self.wins += 1
        else:
            self.losses += 1
        self.total_pnl  += pnl
        self.pnl_sq_sum += pnl ** 2

    def to_dict(self) -> dict:
        return {
            "trades":   self.trade_count,
            "win_rate": round(self.win_rate, 3),
            "avg_pnl":  round(self.avg_pnl, 4),
            "total_pnl": round(self.total_pnl, 4),
            "sharpe":   round(self.sharpe, 3),
        }


class PortfolioRisk:
    """
    Single source of truth for all risk decisions.
    Thread-safe via asyncio.Lock.
    """

    def __init__(self, initial_capital: float):
        self._capital        = initial_capital
        self._initial_capital = initial_capital
        self._peak_capital   = initial_capital
        self._daily_start    = initial_capital
        self._daily_reset    = time.time()
        self._open_trades:   dict[str, TradeRecord]    = {}
        self._closed_trades: list[TradeRecord]         = []
        self._strategy_metrics: dict[str, StrategyMetrics] = {}
        self._lock           = asyncio.Lock()
        self._halted         = False
        self._halt_reason    = ""

    # ── Halt controls ─────────────────────────────────────────────────────────

    def halt(self, reason: str):
        self._halted      = True
        self._halt_reason = reason
        logger.error(f"[RISK] HALTED: {reason}")

    def resume(self):
        self._halted      = False
        self._halt_reason = ""
        logger.info("[RISK] Resumed trading")

    @property
    def is_halted(self) -> bool:
        return self._halted

    # ── Checks ────────────────────────────────────────────────────────────────

    async def can_open(self, symbol: str) -> tuple[bool, str]:
        async with self._lock:
            if self._halted:
                return False, f"halted:{self._halt_reason}"

            if symbol in self._open_trades:
                return False, f"already_in_position:{symbol}"

            if len(self._open_trades) >= config.MAX_POSITIONS:
                return False, f"max_positions:{config.MAX_POSITIONS}"

            # Need at least TRADE_SIZE_PCT of capital available as margin
            min_margin = self._capital * config.TRADE_SIZE_PCT * 0.5
            if self._capital < min_margin:
                return False, f"insufficient_capital:{self._capital:.2f}"

            daily_loss_pct = (self._daily_start - self._capital) / self._daily_start * 100
            if daily_loss_pct >= config.MAX_DAILY_LOSS_PCT:
                self.halt(f"daily_loss_limit:{daily_loss_pct:.1f}%")
                return False, self._halt_reason

            drawdown_pct = (self._peak_capital - self._capital) / self._peak_capital * 100
            if drawdown_pct >= config.MAX_DRAWDOWN_PCT:
                self.halt(f"max_drawdown:{drawdown_pct:.1f}%")
                return False, self._halt_reason

            return True, "ok"

    def should_stop_loss(self, symbol: str, current_price: float) -> bool:
        t = self._open_trades.get(symbol)
        if not t:
            return False
        # Update trailing stop if active
        self._update_trailing(t, current_price)
        stop = t.trail_stop if t.trail_active else t.stop_loss
        return (t.side == "BUY"  and current_price <= stop) or \
               (t.side == "SELL" and current_price >= stop)

    def should_take_profit(self, symbol: str, current_price: float) -> bool:
        t = self._open_trades.get(symbol)
        if not t:
            return False
        return (t.side == "BUY"  and current_price >= t.take_profit) or \
               (t.side == "SELL" and current_price <= t.take_profit)

    def _update_trailing(self, t: "TradeRecord", price: float):
        """Activate and update trailing stop once profit threshold is hit."""
        trail_pct = getattr(config, "TRAILING_STOP_PCT", 1.0) / 100
        if t.side == "BUY":
            if price > t.peak_price:
                t.peak_price = price
            profit_pct = (price - t.entry_price) / t.entry_price
            if profit_pct >= trail_pct:
                t.trail_active = True
                new_trail = t.peak_price * (1 - trail_pct)
                if new_trail > t.trail_stop:
                    t.trail_stop = new_trail
        else:  # SELL
            if t.peak_price == 0 or price < t.peak_price:
                t.peak_price = price
            profit_pct = (t.entry_price - price) / t.entry_price
            if profit_pct >= trail_pct:
                t.trail_active = True
                new_trail = t.peak_price * (1 + trail_pct)
                if t.trail_stop == 0 or new_trail < t.trail_stop:
                    t.trail_stop = new_trail

    # ── Position sizing ───────────────────────────────────────────────────────

    def size_position(
        self,
        price: float,
        kelly_f: float,
        realized_vol: Optional[float] = None,
    ) -> float:
        """
        Kelly-sized position with volatility scaling and leverage.
        Notional = margin × leverage. Returns quantity = notional / price.
        """
        base_f = kelly_f or config.TRADE_SIZE_PCT

        # Volatility adjustment
        if realized_vol is not None and realized_vol > 0:
            if realized_vol > config.VOL_SCALE_HIGH:
                scale = max(0.5, config.VOL_SCALE_HIGH / realized_vol)
                base_f *= scale
            elif realized_vol < config.VOL_SCALE_LOW:
                scale = min(1.25, config.VOL_SCALE_LOW / realized_vol)
                base_f *= scale

        leverage = getattr(config, "LEVERAGE", 1)
        margin   = self._capital * min(base_f, config.TRADE_SIZE_PCT * 2)
        notional = margin * leverage
        return notional / price if price > 0 else 0.0

    # ── Lifecycle ─────────────────────────────────────────────────────────────

    async def record_open(
        self,
        symbol: str,
        side: str,
        price: float,
        qty: float,
        strategy: str = "unknown",
    ):
        leverage = getattr(config, "LEVERAGE", 1)
        # Price needs to move SL_PCT/leverage to cause SL_PCT account loss
        sl_price_pct = config.STOP_LOSS_PCT  / leverage
        tp_price_pct = config.TAKE_PROFIT_PCT / leverage
        sl_mult = (1 - sl_price_pct / 100) if side == "BUY" else (1 + sl_price_pct / 100)
        tp_mult = (1 + tp_price_pct / 100) if side == "BUY" else (1 - tp_price_pct / 100)
        margin  = (price * qty) / leverage  # actual capital locked

        async with self._lock:
            self._open_trades[symbol] = TradeRecord(
                symbol=symbol, side=side, entry_price=price, quantity=qty,
                margin=margin,
                stop_loss=price * sl_mult, take_profit=price * tp_mult,
                leverage=leverage,
                strategy=strategy,
                peak_price=price,
            )
            # Lock margin from available capital
            self._capital -= margin
        logger.info(
            f"[RISK] Opened {side} {qty:.6f} {symbol} @ {price:.4f} "
            f"SL={price*sl_mult:.4f} TP={price*tp_mult:.4f} "
            f"leverage={leverage}x margin={margin:.2f} via {strategy}"
        )

    async def record_close(self, symbol: str, exit_price: float) -> float:
        async with self._lock:
            t = self._open_trades.pop(symbol, None)
            if not t:
                return 0.0
            mult = 1 if t.side == "BUY" else -1
            pnl  = mult * (exit_price - t.entry_price) * t.quantity
            t.exit_price   = exit_price
            t.pnl          = pnl
            t.hold_seconds = time.time() - t.entry_time
            t.open         = False
            self._closed_trades.append(t)
            # Return margin + PnL to capital
            self._capital += t.margin + pnl
            self._peak_capital = max(self._peak_capital, self._capital)
            self._reset_daily_if_needed()

            # Update per-strategy metrics
            if t.strategy not in self._strategy_metrics:
                self._strategy_metrics[t.strategy] = StrategyMetrics()
            self._strategy_metrics[t.strategy].record(pnl)

        logger.info(
            f"[RISK] Closed {t.side} {symbol} @ {exit_price:.4f} | "
            f"PnL={pnl:+.4f} USDT | Capital={self._capital:.2f} | "
            f"Hold={t.hold_seconds/60:.1f}min | Strategy={t.strategy}"
        )
        return pnl

    def _reset_daily_if_needed(self):
        if time.time() - self._daily_reset > 86400:
            self._daily_start = self._capital
            self._daily_reset = time.time()
            logger.info(f"[RISK] Daily reset — new baseline: ${self._capital:.2f}")

    # ── Reporting ─────────────────────────────────────────────────────────────

    @property
    def capital(self) -> float:
        return self._capital

    @property
    def open_positions(self) -> dict:
        return dict(self._open_trades)

    def unrealized_pnl(self, prices: dict[str, float]) -> float:
        """Calculate unrealized PnL across all open positions."""
        total = 0.0
        for sym, t in self._open_trades.items():
            price = prices.get(sym, t.entry_price)
            mult  = 1 if t.side == "BUY" else -1
            total += mult * (price - t.entry_price) * t.quantity
        return total

    def strategy_performance(self) -> dict:
        return {k: v.to_dict() for k, v in self._strategy_metrics.items()}

    def summary(self) -> dict:
        total_pnl  = sum(t.pnl for t in self._closed_trades)
        win_trades = [t for t in self._closed_trades if t.pnl > 0]
        win_rate   = len(win_trades) / len(self._closed_trades) if self._closed_trades else 0
        locked_margin = sum(t.margin for t in self._open_trades.values())
        equity     = self._capital + locked_margin
        drawdown   = (self._peak_capital - equity) / self._peak_capital * 100
        avg_hold   = (
            sum(t.hold_seconds for t in self._closed_trades) / len(self._closed_trades) / 60
            if self._closed_trades else 0
        )
        return {
            "capital":      round(self._capital, 2),
            "equity":       round(equity, 2),
            "total_pnl":    round(total_pnl, 4),
            "trade_count":  len(self._closed_trades),
            "win_rate":     round(win_rate, 3),
            "open_count":   len(self._open_trades),
            "drawdown_pct": round(max(0, drawdown), 2),
            "avg_hold_min": round(avg_hold, 1),
            "halted":       self._halted,
            "halt_reason":  self._halt_reason,
            "leverage":     getattr(config, "LEVERAGE", 1),
        }
