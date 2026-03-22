"""
execution/router.py — Smart order router with:
  - Limit orders inside spread (cheaper fills)
  - Market order fallback for force trades
  - Order timeout (cancel stale limits after LIMIT_ORDER_TIMEOUT seconds)
  - Volatility-adjusted position sizing
  - Trade attribution passed to risk manager
  - Circuit breaker integration
"""

from __future__ import annotations

import asyncio
import time
import numpy as np
import pandas as pd

import config
from exchange.base import Order, OrderSide, OrderType, AbstractExchange
from execution.risk import PortfolioRisk
from alpha.engine import Signal
from utils.logger import get_logger

logger = get_logger(__name__)

BUY  = "BUY"
SELL = "SELL"
HOLD = "HOLD"


class OrderRouter:

    LIMIT_OFFSET_BPS = 2

    def __init__(self, exchange: AbstractExchange, risk: PortfolioRisk):
        self._ex    = exchange
        self._risk  = risk
        # Track pending limit orders: order_id → (symbol, placed_at, strategy)
        self._pending: dict[str, tuple[str, float, str]] = {}

    # ── Force market order (dashboard override) ───────────────────────────────

    async def process_market(self, signal: Signal) -> bool:
        symbol = signal.symbol
        allowed, reason = await self._risk.can_open(symbol)
        if not allowed:
            logger.info(f"[ROUTER] Force blocked: {reason}")
            return False

        price = await self._ex.get_ticker(symbol)
        qty   = self._risk.size_position(price, signal.kelly_f)
        if qty <= 0:
            return False

        side  = OrderSide.BUY if signal.action == BUY else OrderSide.SELL
        order = Order(symbol=symbol, side=side, type=OrderType.MARKET, quantity=round(qty, 6))
        logger.info(f"[ROUTER] FORCE MARKET {side.value} {qty:.6f} {symbol}")
        filled = await self._ex.place_order(order)
        if filled.status.value in ("FILLED", "PARTIAL"):
            await self._risk.record_open(
                symbol, side.value, filled.avg_price, filled.filled_qty,
                strategy="manual_override"
            )
            return True
        return False

    # ── Normal signal processing ──────────────────────────────────────────────

    async def process(self, signal: Signal, df: pd.DataFrame | None = None) -> bool:
        if signal.action == HOLD:
            return False

        symbol = signal.symbol

        # Exit existing position if needed
        if symbol in self._risk.open_positions:
            pos   = self._risk.open_positions[symbol]
            price = await self._ex.get_ticker(symbol)

            if self._risk.should_stop_loss(symbol, price):
                logger.warning(f"[ROUTER] SL hit {symbol} @ {price:.4f}")
                await self._close_position(symbol, pos, price, "stop_loss")
                return True

            if self._risk.should_take_profit(symbol, price):
                logger.info(f"[ROUTER] TP hit {symbol} @ {price:.4f}")
                await self._close_position(symbol, pos, price, "take_profit")
                return True

            # Signal flip — exit then re-enter below
            if (signal.action == BUY and pos.side == "SELL") or \
               (signal.action == SELL and pos.side == "BUY"):
                await self._close_position(symbol, pos, price, "signal_flip")
            else:
                return False  # already in correct direction

        # Open new position
        allowed, reason = await self._risk.can_open(symbol)
        if not allowed:
            logger.info(f"[ROUTER] Blocked: {reason}")
            return False

        price = await self._ex.get_ticker(symbol)

        # Realized vol for position sizing
        vol = self._realized_vol(df) if df is not None else None

        qty  = self._risk.size_position(price, signal.kelly_f, realized_vol=vol)
        if qty <= 0:
            logger.warning(f"[ROUTER] Zero qty for {symbol}")
            return False

        side = OrderSide.BUY if signal.action == BUY else OrderSide.SELL
        ob   = await self._ex.get_orderbook(symbol, depth=5)
        limit_price = self._limit_price(side, price, ob)

        order = Order(
            symbol=symbol, side=side, type=OrderType.LIMIT,
            quantity=round(qty, 6), price=limit_price,
        )

        logger.info(
            f"[ROUTER] {side.value} {qty:.6f} {symbol} @ {limit_price:.4f} "
            f"score={signal.score:.3f} kelly={signal.kelly_f:.3f} "
            f"vol={vol:.4f}" if vol else
            f"[ROUTER] {side.value} {qty:.6f} {symbol} @ {limit_price:.4f} "
            f"score={signal.score:.3f} kelly={signal.kelly_f:.3f}"
        )

        filled = await self._ex.place_order(order)

        if filled.status.value in ("FILLED", "PARTIAL"):
            await self._risk.record_open(
                symbol, side.value, filled.avg_price or limit_price,
                filled.filled_qty, strategy=signal.reason or "alpha_engine"
            )
            return True

        # Track pending limit order for timeout
        if filled.order_id:
            self._pending[filled.order_id] = (symbol, time.time(), signal.reason or "alpha_engine")

        return False

    # ── Exit checker ──────────────────────────────────────────────────────────

    async def check_exits(self, symbols: list[str]):
        for symbol in list(self._risk.open_positions.keys()):
            if symbol not in symbols:
                continue
            price = await self._ex.get_ticker(symbol)
            pos   = self._risk.open_positions.get(symbol)
            if not pos:
                continue
            if self._risk.should_stop_loss(symbol, price):
                await self._close_position(symbol, pos, price, "stop_loss")
            elif self._risk.should_take_profit(symbol, price):
                await self._close_position(symbol, pos, price, "take_profit")

    async def cancel_timed_out_orders(self):
        """Cancel limit orders that haven't filled within LIMIT_ORDER_TIMEOUT."""
        now = time.time()
        for oid, (symbol, placed_at, _) in list(self._pending.items()):
            if now - placed_at > config.LIMIT_ORDER_TIMEOUT:
                cancelled = await self._ex.cancel_order(symbol, oid)
                if cancelled:
                    logger.info(f"[ROUTER] Cancelled timed-out order {oid} for {symbol}")
                del self._pending[oid]

    # ── Helpers ───────────────────────────────────────────────────────────────

    async def _close_position(self, symbol: str, pos, price: float, reason: str):
        close_side = OrderSide.SELL if pos.side == "BUY" else OrderSide.BUY
        order = Order(
            symbol=symbol, side=close_side,
            type=OrderType.MARKET, quantity=pos.quantity,
        )
        filled = await self._ex.place_order(order)
        exit_price = filled.avg_price or price
        pnl = await self._risk.record_close(symbol, exit_price)
        logger.info(f"[ROUTER] Closed {symbol} reason={reason} pnl={pnl:+.4f}")

    def _limit_price(self, side: OrderSide, mid: float, ob: dict) -> float:
        offset = mid * self.LIMIT_OFFSET_BPS / 10_000
        if side == OrderSide.BUY:
            best_ask = ob["asks"][0][0] if ob.get("asks") else mid
            return round(best_ask - offset, 2)
        else:
            best_bid = ob["bids"][0][0] if ob.get("bids") else mid
            return round(best_bid + offset, 2)

    @staticmethod
    def _realized_vol(df: pd.DataFrame) -> float | None:
        """Annualized realized volatility from recent closes."""
        try:
            closes = df["close"].astype(float).tail(config.VOL_LOOKBACK + 1)
            if len(closes) < 5:
                return None
            log_returns = np.log(closes / closes.shift(1)).dropna()
            # Annualize: 5m candles → 288 per day → 365 days
            return float(log_returns.std() * np.sqrt(288 * 365))
        except Exception:
            return None
