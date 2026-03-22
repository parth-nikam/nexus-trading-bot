# exchange/paper.py — Paper trading engine
# Simulates fills with real Binance prices. Market orders fill immediately.
# Limit orders fill when market price crosses the limit price.

import asyncio
import time
import uuid
import aiohttp
from typing import Optional

import config
from exchange.base import (
    AbstractExchange, Order, OrderSide, OrderType, OrderStatus, Balance, Position
)
from utils.logger import get_logger
from utils.retry import retry

logger = get_logger(__name__)

BINANCE_PUBLIC = "https://api.binance.com"


class PaperExchange(AbstractExchange):
    """
    Paper trading with real Binance price data.
    - MARKET orders fill immediately at current price + slippage
    - LIMIT orders fill when market price crosses the limit
    - Tracks USDT balance and open positions
    - Calls back on_fill when a limit order fills (for risk manager notification)
    """

    FEE_RATE = 0.001  # 0.1% taker fee

    def __init__(self, initial_capital: float = 10_000.0):
        self._capital   = initial_capital
        self._balances  = {"USDT": initial_capital}
        self._positions: dict[str, Position] = {}
        self._orders:    dict[str, Order]    = {}
        self._trade_log: list[dict]          = []
        self._session:   Optional[aiohttp.ClientSession] = None
        # Callback: called when a pending limit order fills
        # signature: (order: Order) -> None
        self.on_limit_fill = None

    async def _session_get(self) -> aiohttp.ClientSession:
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession()
        return self._session

    @retry(max_attempts=3, base_delay=1.0, exceptions=(aiohttp.ClientError, Exception))
    async def _get(self, path: str, params: dict = None, base: str = None) -> dict | list:
        session = await self._session_get()
        url = (base or BINANCE_PUBLIC) + path
        async with session.get(url, params=params, timeout=aiohttp.ClientTimeout(total=10)) as r:
            r.raise_for_status()
            return await r.json()

    # ── Market data ───────────────────────────────────────────────────────────

    async def get_candles(self, symbol: str, interval: str, limit: int) -> list[dict]:
        data = await self._get("/api/v3/klines", {
            "symbol": symbol, "interval": interval, "limit": limit
        })
        candles = [
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
        if len(candles) > 1:
            times = [c["time"] for c in candles]
            diffs = [times[i+1] - times[i] for i in range(len(times)-1)]
            expected = diffs[0]
            gaps = sum(1 for d in diffs if d > expected * 1.5)
            if gaps:
                logger.warning(f"[{symbol}] {gaps} candle gap(s) detected")
        return candles

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
        try:
            data = await self._get("/fapi/v1/premiumIndex", {"symbol": symbol},
                                   base="https://fapi.binance.com")
            return float(data.get("lastFundingRate", 0))
        except Exception:
            return 0.0

    # ── Order management ──────────────────────────────────────────────────────

    async def place_order(self, order: Order) -> Order:
        order.order_id  = str(uuid.uuid4())[:8]
        order.timestamp = time.time()

        price = await self.get_ticker(order.symbol)

        if order.type == OrderType.MARKET:
            slippage   = config.SLIPPAGE_BPS / 10_000
            fill_price = price * (1 + slippage) if order.side == OrderSide.BUY else price * (1 - slippage)
            await self._fill_order(order, fill_price)

        elif order.type == OrderType.LIMIT:
            order.price = order.price or price
            # Immediately fillable?
            if order.side == OrderSide.BUY and price <= order.price * 1.001:
                await self._fill_order(order, min(price, order.price))
            elif order.side == OrderSide.SELL and price >= order.price * 0.999:
                await self._fill_order(order, max(price, order.price))
            else:
                order.status = OrderStatus.OPEN
                self._orders[order.order_id] = order
                logger.info(
                    f"[PAPER] Limit order queued: {order.side.value} "
                    f"{order.quantity} {order.symbol} @ {order.price:.4f} "
                    f"(market={price:.4f})"
                )

        return order

    async def _fill_order(self, order: Order, fill_price: float):
        cost  = fill_price * order.quantity
        fee   = cost * self.FEE_RATE
        usdt  = self._balances.get("USDT", 0)
        asset = order.symbol.replace("USDT", "")

        if order.side == OrderSide.BUY:
            total_cost = cost + fee
            # Check if this is closing a short position
            existing = self._positions.get(order.symbol)
            if existing and existing.side == OrderSide.SELL:
                # Closing short: buy back asset, return borrowed, realize PnL
                if usdt < total_cost:
                    logger.warning(
                        f"[PAPER] Insufficient USDT to close short: need {total_cost:.2f}, have {usdt:.2f}"
                    )
                    order.status = OrderStatus.CANCELLED
                    return
                self._balances["USDT"] = usdt - total_cost
                self._balances[asset]  = max(0.0, self._balances.get(asset, 0) - order.quantity)
                pos = self._positions.pop(order.symbol)
                realized = (pos.entry_price - fill_price) * order.quantity - fee
                logger.info(f"[PAPER] Short closed | PnL={realized:+.4f} USDT")
            else:
                if usdt < total_cost:
                    logger.warning(
                        f"[PAPER] Insufficient USDT: need {total_cost:.2f}, have {usdt:.2f}"
                    )
                    order.status = OrderStatus.CANCELLED
                    return
                self._balances["USDT"] = usdt - total_cost
                self._balances[asset]  = self._balances.get(asset, 0) + order.quantity
                self._positions[order.symbol] = Position(
                    symbol=order.symbol, side=OrderSide.BUY,
                    quantity=order.quantity, entry_price=fill_price,
                    current_price=fill_price,
                )

        else:  # SELL
            asset_bal = self._balances.get(asset, 0)
            if asset_bal < order.quantity * 0.999:  # 0.1% tolerance
                # Try to close from any tracked position
                pos_qty = self._positions.get(order.symbol)
                if pos_qty:
                    order.quantity = pos_qty.quantity
                    asset_bal = order.quantity
                elif config.LEVERAGE > 1:
                    # Leveraged short — borrow the asset (paper trading)
                    # Deduct USDT margin, track short position
                    margin = cost / config.LEVERAGE
                    if self._balances.get("USDT", 0) < margin:
                        logger.warning(
                            f"[PAPER] Insufficient USDT for short margin: "
                            f"need {margin:.2f}, have {self._balances.get('USDT',0):.2f}"
                        )
                        order.status = OrderStatus.CANCELLED
                        return
                    # Borrow asset and immediately sell it
                    self._balances[asset] = order.quantity  # synthetic borrow
                    asset_bal = order.quantity
                else:
                    logger.warning(
                        f"[PAPER] Insufficient {asset}: need {order.quantity:.6f}, "
                        f"have {asset_bal:.6f}"
                    )
                    order.status = OrderStatus.CANCELLED
                    return
            proceeds = cost - fee
            self._balances[asset]  = max(0.0, self._balances.get(asset, 0) - order.quantity)
            self._balances["USDT"] = usdt + proceeds
            if order.symbol in self._positions:
                pos = self._positions.pop(order.symbol)
                realized = (fill_price - pos.entry_price) * order.quantity - fee
                logger.info(f"[PAPER] Position closed | PnL={realized:+.4f} USDT")
            else:
                # Opening a short position
                self._positions[order.symbol] = Position(
                    symbol=order.symbol, side=OrderSide.SELL,
                    quantity=order.quantity, entry_price=fill_price,
                    current_price=fill_price,
                )

        order.status     = OrderStatus.FILLED
        order.filled_qty = order.quantity
        order.avg_price  = fill_price
        order.fee        = fee

        self._trade_log.append({
            "id":        order.order_id,
            "symbol":    order.symbol,
            "side":      order.side.value,
            "qty":       order.quantity,
            "price":     fill_price,
            "fee":       fee,
            "timestamp": order.timestamp,
            "strategy":  getattr(order, "strategy", "unknown"),
        })

        logger.info(
            f"[PAPER] FILLED {order.side.value} {order.quantity:.6f} {order.symbol} "
            f"@ {fill_price:.4f} | fee={fee:.4f} USDT | "
            f"USDT_bal={self._balances['USDT']:.2f}"
        )

    async def cancel_order(self, symbol: str, order_id: str) -> bool:
        if order_id in self._orders:
            self._orders[order_id].status = OrderStatus.CANCELLED
            del self._orders[order_id]
            logger.info(f"[PAPER] Cancelled order {order_id} for {symbol}")
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

    async def check_limit_fills(self):
        """
        Check if any queued limit orders are now fillable at market price.
        Calls on_limit_fill callback so the router can record the open position.
        """
        for oid, order in list(self._orders.items()):
            try:
                price = await self.get_ticker(order.symbol)
                should_fill = (
                    (order.side == OrderSide.BUY  and price <= order.price * 1.002) or
                    (order.side == OrderSide.SELL and price >= order.price * 0.998)
                )
                if should_fill:
                    fill_price = price  # fill at market, not limit (more realistic)
                    await self._fill_order(order, fill_price)
                    del self._orders[oid]
                    # Notify router so it can record_open in risk manager
                    if self.on_limit_fill and order.status == OrderStatus.FILLED:
                        await self.on_limit_fill(order)
            except Exception as e:
                logger.error(f"[PAPER] check_limit_fills error: {e}")

    # ── Portfolio snapshot ────────────────────────────────────────────────────

    async def portfolio_value(self) -> float:
        total = self._balances.get("USDT", 0)
        for symbol, pos in self._positions.items():
            try:
                price = await self.get_ticker(symbol)
                pos.current_price = price
                total += price * pos.quantity
            except Exception:
                total += pos.entry_price * pos.quantity
        return total

    def trade_history(self) -> list[dict]:
        return list(self._trade_log)

    async def close(self):
        if self._session and not self._session.closed:
            await self._session.close()
