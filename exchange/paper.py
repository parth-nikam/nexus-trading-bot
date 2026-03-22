# exchange/paper.py — Paper trading engine
# Zero-latency simulation with realistic slippage model.
# Pulls real market data from Binance public API, simulates fills locally.

import asyncio
import time
import uuid
import aiohttp
from typing import Optional
from collections import defaultdict

import config
from exchange.base import (
    AbstractExchange, Order, OrderSide, OrderType, OrderStatus, Balance, Position
)
from utils.logger import get_logger

logger = get_logger(__name__)

BINANCE_PUBLIC = "https://api.binance.com"
BINANCE_TESTNET_PUBLIC = "https://testnet.binance.vision"


class PaperExchange(AbstractExchange):
    """
    Simulates exchange with real price data.
    Fills LIMIT orders when market price crosses the limit.
    Applies configurable slippage to MARKET orders.
    Tracks P&L, fees, and balance in memory.
    """

    FEE_RATE = 0.001  # 0.1% taker fee

    def __init__(self, initial_capital: float = 10_000.0):
        self._capital   = initial_capital
        self._balances  = {"USDT": initial_capital}
        self._positions: dict[str, Position] = {}
        self._orders:    dict[str, Order]    = {}
        self._trade_log: list[dict]          = []
        self._session:   Optional[aiohttp.ClientSession] = None
        self._base_url   = BINANCE_PUBLIC

    async def _session_get(self) -> aiohttp.ClientSession:
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession()
        return self._session

    async def _get(self, path: str, params: dict = None) -> dict | list:
        session = await self._session_get()
        async with session.get(f"{self._base_url}{path}", params=params) as r:
            r.raise_for_status()
            return await r.json()

    # ── Market data ───────────────────────────────────────────────────────────

    async def get_candles(self, symbol: str, interval: str, limit: int) -> list[dict]:
        data = await self._get("/api/v3/klines", {
            "symbol": symbol, "interval": interval, "limit": limit
        })
        return [
            {
                "time":   row[0],
                "open":   float(row[1]),
                "high":   float(row[2]),
                "low":    float(row[3]),
                "close":  float(row[4]),
                "volume": float(row[5]),
            }
            for row in data
        ]

    async def get_ticker(self, symbol: str) -> float:
        data = await self._get("/api/v3/ticker/price", {"symbol": symbol})
        return float(data["price"])

    async def get_orderbook(self, symbol: str, depth: int = 20) -> dict:
        data = await self._get("/api/v3/depth", {"symbol": symbol, "limit": depth})
        return {
            "bids": [(float(p), float(q)) for p, q in data["bids"]],
            "asks": [(float(p), float(q)) for p, q in data["asks"]],
        }

    async def get_funding_rate(self, symbol: str) -> float:
        """Binance perpetual funding rate — useful for carry signals."""
        try:
            data = await self._get("/fapi/v1/premiumIndex", {"symbol": symbol})
            return float(data.get("lastFundingRate", 0))
        except Exception:
            return 0.0

    # ── Order management ──────────────────────────────────────────────────────

    async def place_order(self, order: Order) -> Order:
        order.order_id = str(uuid.uuid4())[:8]
        order.timestamp = time.time()

        price = await self.get_ticker(order.symbol)

        if order.type == OrderType.MARKET:
            # Apply slippage
            slippage = config.SLIPPAGE_BPS / 10_000
            fill_price = price * (1 + slippage) if order.side == OrderSide.BUY else price * (1 - slippage)
            await self._fill_order(order, fill_price)

        elif order.type == OrderType.LIMIT:
            order.price = order.price or price
            # Check if immediately fillable
            if order.side == OrderSide.BUY and price <= order.price:
                await self._fill_order(order, order.price)
            elif order.side == OrderSide.SELL and price >= order.price:
                await self._fill_order(order, order.price)
            else:
                order.status = OrderStatus.OPEN
                self._orders[order.order_id] = order
                logger.info(f"[PAPER] Limit order queued: {order.side} {order.quantity} {order.symbol} @ {order.price:.4f}")

        return order

    async def _fill_order(self, order: Order, fill_price: float):
        cost  = fill_price * order.quantity
        fee   = cost * self.FEE_RATE
        usdt  = self._balances.get("USDT", 0)
        asset = order.symbol.replace("USDT", "")

        if order.side == OrderSide.BUY:
            total_cost = cost + fee
            if usdt < total_cost:
                logger.warning(f"[PAPER] Insufficient USDT: need {total_cost:.2f}, have {usdt:.2f}")
                order.status = OrderStatus.CANCELLED
                return
            self._balances["USDT"] = usdt - total_cost
            self._balances[asset]  = self._balances.get(asset, 0) + order.quantity
            self._positions[order.symbol] = Position(
                symbol=order.symbol, side=OrderSide.BUY,
                quantity=order.quantity, entry_price=fill_price,
                current_price=fill_price,
            )
        else:
            asset_bal = self._balances.get(asset, 0)
            if asset_bal < order.quantity:
                logger.warning(f"[PAPER] Insufficient {asset}: need {order.quantity:.6f}, have {asset_bal:.6f}")
                order.status = OrderStatus.CANCELLED
                return
            proceeds = cost - fee
            self._balances[asset]  = asset_bal - order.quantity
            self._balances["USDT"] = usdt + proceeds
            if order.symbol in self._positions:
                pos = self._positions.pop(order.symbol)
                realized = (fill_price - pos.entry_price) * order.quantity - fee
                logger.info(f"[PAPER] Position closed | PnL={realized:+.4f} USDT")

        order.status    = OrderStatus.FILLED
        order.filled_qty = order.quantity
        order.avg_price  = fill_price
        order.fee        = fee

        self._trade_log.append({
            "id":         order.order_id,
            "symbol":     order.symbol,
            "side":       order.side.value,
            "qty":        order.quantity,
            "price":      fill_price,
            "fee":        fee,
            "timestamp":  order.timestamp,
        })

        logger.info(
            f"[PAPER] FILLED {order.side.value} {order.quantity:.6f} {order.symbol} "
            f"@ {fill_price:.4f} | fee={fee:.4f} USDT"
        )

    async def cancel_order(self, symbol: str, order_id: str) -> bool:
        if order_id in self._orders:
            self._orders[order_id].status = OrderStatus.CANCELLED
            del self._orders[order_id]
            return True
        return False

    async def get_balances(self) -> list[Balance]:
        return [
            Balance(asset=k, free=v, locked=0.0)
            for k, v in self._balances.items()
            if v > 0
        ]

    async def get_open_orders(self, symbol: str) -> list[Order]:
        return [o for o in self._orders.values() if o.symbol == symbol]

    # ── Portfolio snapshot ────────────────────────────────────────────────────

    async def portfolio_value(self) -> float:
        """Total portfolio value in USDT."""
        total = self._balances.get("USDT", 0)
        for symbol, pos in self._positions.items():
            price = await self.get_ticker(symbol)
            pos.update_price(price)
            total += pos.current_price * pos.quantity
        return total

    def trade_history(self) -> list[dict]:
        return list(self._trade_log)

    async def close(self):
        if self._session and not self._session.closed:
            await self._session.close()
