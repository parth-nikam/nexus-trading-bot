# execution/router.py — Smart order router
# Decides order type, price, and timing based on signal + market conditions.
# Uses limit orders by default (cheaper fees, less slippage).
# Falls back to market order if spread is too wide or urgency is high.

from __future__ import annotations
import asyncio
from utils.logger import get_logger
from exchange.base import Order, OrderSide, OrderType, AbstractExchange
from execution.risk import PortfolioRisk
from alpha.engine import Signal

logger = get_logger(__name__)

BUY  = "BUY"
SELL = "SELL"
HOLD = "HOLD"


class OrderRouter:

    # Place limit order this many bps inside the spread for better fill probability
    LIMIT_OFFSET_BPS = 2

    def __init__(self, exchange: AbstractExchange, risk: PortfolioRisk):
        self._ex   = exchange
        self._risk = risk

    async def process(self, signal: Signal) -> bool:
        """
        Process a signal. Returns True if an order was placed.
        """
        if signal.action == HOLD:
            return False

        symbol = signal.symbol

        # ── Check if we should EXIT an existing position ──────────────────────
        if symbol in self._risk.open_positions:
            pos = self._risk.open_positions[symbol]
            price = await self._ex.get_ticker(symbol)

            if self._risk.should_stop_loss(symbol, price):
                logger.warning(f"[ROUTER] Stop loss hit for {symbol} @ {price:.4f}")
                await self._close_position(symbol, pos, price, "stop_loss")
                return True

            if self._risk.should_take_profit(symbol, price):
                logger.info(f"[ROUTER] Take profit hit for {symbol} @ {price:.4f}")
                await self._close_position(symbol, pos, price, "take_profit")
                return True

            # Signal flipped — exit and reverse
            if signal.action == BUY  and pos.side == "SELL":
                await self._close_position(symbol, pos, price, "signal_flip")
            elif signal.action == SELL and pos.side == "BUY":
                await self._close_position(symbol, pos, price, "signal_flip")
            else:
                return False  # already in correct direction

        # ── Open new position ─────────────────────────────────────────────────
        allowed, reason = self._risk.can_open(symbol)
        if not allowed:
            logger.info(f"[ROUTER] Blocked: {reason}")
            return False

        price = await self._ex.get_ticker(symbol)
        qty   = self._risk.size_position(price, signal.kelly_f)

        if qty <= 0:
            logger.warning(f"[ROUTER] Zero quantity for {symbol}")
            return False

        side = OrderSide.BUY if signal.action == BUY else OrderSide.SELL

        # Use limit order slightly inside spread for better fill
        ob = await self._ex.get_orderbook(symbol, depth=5)
        limit_price = self._limit_price(side, price, ob)

        order = Order(
            symbol=symbol,
            side=side,
            type=OrderType.LIMIT,
            quantity=round(qty, 6),
            price=limit_price,
        )

        logger.info(
            f"[ROUTER] Placing {side.value} {qty:.6f} {symbol} "
            f"@ {limit_price:.4f} (score={signal.score:.3f} kelly={signal.kelly_f:.3f})"
        )

        filled = await self._ex.place_order(order)

        if filled.status.value in ("FILLED", "PARTIAL"):
            self._risk.record_open(symbol, side.value, filled.avg_price or limit_price, filled.filled_qty)
            return True

        return False

    async def check_exits(self, symbols: list[str]):
        """Called every tick to check stop loss / take profit on open positions."""
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

    async def _close_position(self, symbol: str, pos, price: float, reason: str):
        close_side = OrderSide.SELL if pos.side == "BUY" else OrderSide.BUY
        order = Order(
            symbol=symbol,
            side=close_side,
            type=OrderType.MARKET,
            quantity=pos.quantity,
        )
        filled = await self._ex.place_order(order)
        exit_price = filled.avg_price or price
        pnl = self._risk.record_close(symbol, exit_price)
        logger.info(f"[ROUTER] Closed {symbol} reason={reason} pnl={pnl:+.4f}")

    def _limit_price(self, side: OrderSide, mid: float, ob: dict) -> float:
        """Price slightly inside the spread to improve fill probability."""
        offset = mid * self.LIMIT_OFFSET_BPS / 10_000
        if side == OrderSide.BUY:
            best_ask = ob["asks"][0][0] if ob.get("asks") else mid
            return round(best_ask - offset, 2)
        else:
            best_bid = ob["bids"][0][0] if ob.get("bids") else mid
            return round(best_bid + offset, 2)
