"""
execution/router.py — Smart order router.

Paper trading: uses MARKET orders (always fill, no stuck limit orders).
Live trading: uses LIMIT orders inside spread for better fills.
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
        # pending limit orders: order_id → (symbol, placed_at, strategy)
        self._pending: dict[str, tuple[str, float, str]] = {}

        # Wire up limit fill callback on paper exchange
        if hasattr(exchange, "on_limit_fill"):
            exchange.on_limit_fill = self._on_limit_filled

    async def _on_limit_filled(self, order: Order):
        """Called by PaperExchange when a queued limit order fills."""
        symbol   = order.symbol
        strategy = self._pending.pop(order.order_id, (None, None, "alpha_engine"))[2]
        if symbol not in self._risk.open_positions:
            await self._risk.record_open(
                symbol, order.side.value, order.avg_price,
                order.filled_qty, strategy=strategy
            )
            logger.info(
                f"[ROUTER] Limit filled → position opened: "
                f"{order.side.value} {order.filled_qty:.6f} {symbol} @ {order.avg_price:.4f}"
            )

    # ── Force market order (dashboard override) ───────────────────────────────

    async def process_market(self, signal: Signal) -> bool:
        symbol = signal.symbol

        # Close existing opposite position first
        if symbol in self._risk.open_positions:
            pos = self._risk.open_positions[symbol]
            if (signal.action == BUY and pos.side == "SELL") or \
               (signal.action == SELL and pos.side == "BUY"):
                price = await self._ex.get_ticker(symbol)
                await self._close_position(symbol, pos, price, "signal_flip")

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
        logger.info(f"[ROUTER] FORCE MARKET {side.value} {qty:.6f} {symbol} @ ~{price:.2f}")
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

        # Check exits on open position
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

            # Signal flip
            if (signal.action == BUY and pos.side == "SELL") or \
               (signal.action == SELL and pos.side == "BUY"):
                await self._close_position(symbol, pos, price, "signal_flip")
            else:
                return False  # already in correct direction

        # Don't open if we already have a pending limit order for this symbol
        pending_symbols = {v[0] for v in self._pending.values()}
        if symbol in pending_symbols:
            logger.debug(f"[ROUTER] Skipping {symbol} — limit order already pending")
            return False

        allowed, reason = await self._risk.can_open(symbol)
        if not allowed:
            logger.info(f"[ROUTER] Blocked: {reason}")
            return False

        price = await self._ex.get_ticker(symbol)
        vol   = self._realized_vol(df) if df is not None else None
        qty   = self._risk.size_position(price, signal.kelly_f, realized_vol=vol)

        if qty <= 0:
            logger.warning(f"[ROUTER] Zero qty for {symbol}")
            return False

        side = OrderSide.BUY if signal.action == BUY else OrderSide.SELL

        # Paper trading: use MARKET orders — always fill, no stuck orders
        # Live trading: use LIMIT orders inside spread for better fills
        if config.PAPER_TRADING:
            order_type  = OrderType.MARKET
            limit_price = None
        else:
            order_type  = OrderType.LIMIT
            ob          = await self._ex.get_orderbook(symbol, depth=5)
            limit_price = self._limit_price(side, price, ob)

        order = Order(
            symbol=symbol, side=side, type=order_type,
            quantity=round(qty, 6), price=limit_price,
        )

        vol_str = f" vol={vol:.4f}" if vol else ""
        logger.info(
            f"[ROUTER] {side.value} {qty:.6f} {symbol} @ "
            f"{'MARKET~' + str(round(price, 2)) if order_type == OrderType.MARKET else str(limit_price)} "
            f"score={signal.score:.3f} kelly={signal.kelly_f:.3f}{vol_str}"
        )

        filled = await self._ex.place_order(order)

        if filled.status.value in ("FILLED", "PARTIAL"):
            await self._risk.record_open(
                symbol, side.value,
                filled.avg_price or price,
                filled.filled_qty,
                strategy=signal.reason.split("|")[0].strip() if signal.reason else "alpha_engine"
            )
            logger.info(
                f"[ROUTER] Position opened: {side.value} {filled.filled_qty:.6f} "
                f"{symbol} @ {filled.avg_price:.4f}"
            )
            return True

        # Limit order queued (live trading)
        if filled.order_id and filled.status.value == "OPEN":
            strategy = signal.reason.split("|")[0].strip() if signal.reason else "alpha_engine"
            self._pending[filled.order_id] = (symbol, time.time(), strategy)

        return False

    # ── Exit checker ──────────────────────────────────────────────────────────

    async def check_exits(self, symbols: list[str]):
        for symbol in list(self._risk.open_positions.keys()):
            if symbol not in symbols:
                continue
            try:
                price = await self._ex.get_ticker(symbol)
                pos   = self._risk.open_positions.get(symbol)
                if not pos:
                    continue
                if self._risk.should_stop_loss(symbol, price):
                    logger.warning(f"[ROUTER] SL triggered {symbol} @ {price:.4f}")
                    await self._close_position(symbol, pos, price, "stop_loss")
                elif self._risk.should_take_profit(symbol, price):
                    logger.info(f"[ROUTER] TP triggered {symbol} @ {price:.4f}")
                    await self._close_position(symbol, pos, price, "take_profit")
            except Exception as e:
                logger.error(f"[ROUTER] check_exits error {symbol}: {e}")

    async def cancel_timed_out_orders(self):
        """Cancel limit orders that haven't filled within LIMIT_ORDER_TIMEOUT."""
        now = time.time()
        for oid, (symbol, placed_at, _) in list(self._pending.items()):
            if now - placed_at > config.LIMIT_ORDER_TIMEOUT:
                cancelled = await self._ex.cancel_order(symbol, oid)
                if cancelled:
                    logger.info(f"[ROUTER] Cancelled timed-out order {oid} for {symbol}")
                self._pending.pop(oid, None)

    # ── Helpers ───────────────────────────────────────────────────────────────

    async def _close_position(self, symbol: str, pos, price: float, reason: str):
        """Close an open position with a market order."""
        close_side = OrderSide.SELL if pos.side == "BUY" else OrderSide.BUY
        order = Order(
            symbol=symbol, side=close_side,
            type=OrderType.MARKET, quantity=round(pos.quantity, 6),
        )
        filled = await self._ex.place_order(order)
        exit_price = filled.avg_price or price
        pnl = await self._risk.record_close(symbol, exit_price)
        logger.info(
            f"[ROUTER] Closed {symbol} reason={reason} "
            f"exit={exit_price:.4f} pnl={pnl:+.4f} USDT"
        )

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
        try:
            closes = df["close"].astype(float).tail(config.VOL_LOOKBACK + 1)
            if len(closes) < 5:
                return None
            log_returns = np.log(closes / closes.shift(1)).dropna()
            return float(log_returns.std() * np.sqrt(288 * 365))
        except Exception:
            return None
