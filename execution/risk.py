# execution/risk.py — Portfolio-level risk manager
# Enforces: max positions, daily loss limit, max drawdown, per-trade stop/TP.

from __future__ import annotations
import time
from dataclasses import dataclass, field
from utils.logger import get_logger
import config

logger = get_logger(__name__)


@dataclass
class TradeRecord:
    symbol:      str
    side:        str
    entry_price: float
    quantity:    float
    stop_loss:   float
    take_profit: float
    entry_time:  float = field(default_factory=time.time)
    exit_price:  float = 0.0
    pnl:         float = 0.0
    open:        bool  = True


class PortfolioRisk:
    """
    Single source of truth for all risk decisions.
    The execution router asks this before placing any order.
    """

    def __init__(self, initial_capital: float):
        self._capital       = initial_capital
        self._peak_capital  = initial_capital
        self._daily_start   = initial_capital
        self._daily_reset   = time.time()
        self._open_trades:  dict[str, TradeRecord] = {}
        self._closed_trades: list[TradeRecord]     = []

    # ── Checks ────────────────────────────────────────────────────────────────

    def can_open(self, symbol: str) -> tuple[bool, str]:
        """Returns (allowed, reason)."""
        if symbol in self._open_trades:
            return False, f"already_in_position:{symbol}"

        if len(self._open_trades) >= config.MAX_POSITIONS:
            return False, f"max_positions:{config.MAX_POSITIONS}"

        daily_loss_pct = (self._daily_start - self._capital) / self._daily_start * 100
        if daily_loss_pct >= config.MAX_DAILY_LOSS_PCT:
            return False, f"daily_loss_limit:{daily_loss_pct:.1f}%"

        drawdown_pct = (self._peak_capital - self._capital) / self._peak_capital * 100
        if drawdown_pct >= config.MAX_DRAWDOWN_PCT:
            return False, f"max_drawdown:{drawdown_pct:.1f}%"

        return True, "ok"

    def should_stop_loss(self, symbol: str, current_price: float) -> bool:
        t = self._open_trades.get(symbol)
        if not t:
            return False
        if t.side == "BUY"  and current_price <= t.stop_loss:
            return True
        if t.side == "SELL" and current_price >= t.stop_loss:
            return True
        return False

    def should_take_profit(self, symbol: str, current_price: float) -> bool:
        t = self._open_trades.get(symbol)
        if not t:
            return False
        if t.side == "BUY"  and current_price >= t.take_profit:
            return True
        if t.side == "SELL" and current_price <= t.take_profit:
            return True
        return False

    # ── Position sizing ───────────────────────────────────────────────────────

    def size_position(self, price: float, kelly_f: float) -> float:
        """Returns quantity to buy/sell based on Kelly fraction of capital."""
        usable = self._capital * kelly_f
        return usable / price

    # ── Lifecycle ─────────────────────────────────────────────────────────────

    def record_open(self, symbol: str, side: str, price: float, qty: float):
        sl_mult = 1 - config.STOP_LOSS_PCT / 100 if side == "BUY" else 1 + config.STOP_LOSS_PCT / 100
        tp_mult = 1 + config.TAKE_PROFIT_PCT / 100 if side == "BUY" else 1 - config.TAKE_PROFIT_PCT / 100
        self._open_trades[symbol] = TradeRecord(
            symbol=symbol, side=side, entry_price=price, quantity=qty,
            stop_loss=price * sl_mult, take_profit=price * tp_mult,
        )
        logger.info(
            f"[RISK] Opened {side} {qty:.6f} {symbol} @ {price:.4f} "
            f"SL={price*sl_mult:.4f} TP={price*tp_mult:.4f}"
        )

    def record_close(self, symbol: str, exit_price: float) -> float:
        t = self._open_trades.pop(symbol, None)
        if not t:
            return 0.0
        mult = 1 if t.side == "BUY" else -1
        pnl  = mult * (exit_price - t.entry_price) * t.quantity
        t.exit_price = exit_price
        t.pnl        = pnl
        t.open       = False
        self._closed_trades.append(t)
        self._capital += pnl
        self._peak_capital = max(self._peak_capital, self._capital)
        self._reset_daily_if_needed()
        logger.info(
            f"[RISK] Closed {t.side} {symbol} @ {exit_price:.4f} | "
            f"PnL={pnl:+.4f} USDT | Capital={self._capital:.2f}"
        )
        return pnl

    def _reset_daily_if_needed(self):
        if time.time() - self._daily_reset > 86400:
            self._daily_start = self._capital
            self._daily_reset = time.time()

    # ── Reporting ─────────────────────────────────────────────────────────────

    @property
    def capital(self) -> float:
        return self._capital

    @property
    def open_positions(self) -> dict:
        return dict(self._open_trades)

    def summary(self) -> dict:
        total_pnl  = sum(t.pnl for t in self._closed_trades)
        win_trades = [t for t in self._closed_trades if t.pnl > 0]
        win_rate   = len(win_trades) / len(self._closed_trades) if self._closed_trades else 0
        drawdown   = (self._peak_capital - self._capital) / self._peak_capital * 100
        return {
            "capital":      round(self._capital, 2),
            "total_pnl":    round(total_pnl, 4),
            "trade_count":  len(self._closed_trades),
            "win_rate":     round(win_rate, 3),
            "open_count":   len(self._open_trades),
            "drawdown_pct": round(drawdown, 2),
        }
