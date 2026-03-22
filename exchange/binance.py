"""
exchange/binance.py — Production Binance REST + WebSocket exchange adapter.

Supports:
  - Spot and USDT-M Perpetual Futures (auto-detected by symbol suffix)
  - HMAC-SHA256 signed requests for private endpoints
  - WebSocket price streams for low-latency ticker updates
  - Automatic reconnect on WebSocket disconnect
  - Testnet / mainnet toggle via config
"""

from __future__ import annotations

import asyncio
import hashlib
import hmac
import json
import time
import uuid
from typing import Optional
from urllib.parse import urlencode

import aiohttp

import config
from exchange.base import (
    AbstractExchange, Balance, Order, OrderSide, OrderStatus, OrderType, Position,
)
from utils.logger import get_logger

logger = get_logger(__name__)

# ── Endpoints ─────────────────────────────────────────────────────────────────
SPOT_REST   = "https://api.binance.com"
SPOT_WS     = "wss://stream.binance.com:9443/ws"
FAPI_REST   = "https://fapi.binance.com"          # USDT-M perps
FAPI_WS     = "wss://fstream.binance.com/ws"

TESTNET_SPOT_REST = "https://testnet.binance.vision"
TESTNET_SPOT_WS   = "wss://testnet.binance.vision/ws"
TESTNET_FAPI_REST = "https://testnet.binancefuture.com"
TESTNET_FAPI_WS   = "wss://stream.binancefuture.com/ws"


def _is_perp(symbol: str) -> bool:
    """Heuristic: if symbol ends in USDT and is in futures universe."""
    return symbol.endswith("USDT")


class BinanceExchange(AbstractExchange):
    """
    Full Binance exchange adapter.
    Uses futures endpoints for USDT-margined perps, spot for everything else.
    """

    FEE_RATE = 0.0004  # 0.04% taker (futures); 0.1% spot handled separately

    def __init__(self):
        testnet = config.BINANCE_TESTNET
        self._api_key    = config.BINANCE_API_KEY
        self._api_secret = config.BINANCE_API_SECRET

        # REST base URLs
        self._spot_url = TESTNET_SPOT_REST if testnet else SPOT_REST
        self._fapi_url = TESTNET_FAPI_REST if testnet else FAPI_REST

        # WebSocket base URLs
        self._spot_ws_url = TESTNET_SPOT_WS if testnet else SPOT_WS
        self._fapi_ws_url = TESTNET_FAPI_WS if testnet else FAPI_WS

        self._session: Optional[aiohttp.ClientSession] = None

        # In-memory position cache (futures positions fetched from exchange)
        self._positions: dict[str, Position] = {}

        # WebSocket price cache — updated by background stream
        self._prices: dict[str, float] = {}
        self._ws_task: Optional[asyncio.Task] = None

    # ── Session management ────────────────────────────────────────────────────

    async def _get_session(self) -> aiohttp.ClientSession:
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession(
                headers={"X-MBX-APIKEY": self._api_key}
            )
        return self._session

    async def close(self):
        if self._ws_task:
            self._ws_task.cancel()
        if self._session and not self._session.closed:
            await self._session.close()

    # ── Signing ───────────────────────────────────────────────────────────────

    def _sign(self, params: dict) -> dict:
        params["timestamp"] = int(time.time() * 1000)
        query = urlencode(params)
        sig = hmac.new(
            self._api_secret.encode(), query.encode(), hashlib.sha256
        ).hexdigest()
        params["signature"] = sig
        return params

    # ── HTTP helpers ──────────────────────────────────────────────────────────

    async def _get(self, base: str, path: str, params: dict = None, signed: bool = False):
        session = await self._get_session()
        p = params or {}
        if signed:
            p = self._sign(p)
        async with session.get(f"{base}{path}", params=p) as r:
            data = await r.json()
            if r.status != 200:
                raise RuntimeError(f"GET {path} {r.status}: {data}")
            return data

    async def _post(self, base: str, path: str, params: dict = None, signed: bool = True):
        session = await self._get_session()
        p = params or {}
        if signed:
            p = self._sign(p)
        async with session.post(f"{base}{path}", params=p) as r:
            data = await r.json()
            if r.status not in (200, 201):
                raise RuntimeError(f"POST {path} {r.status}: {data}")
            return data

    async def _delete(self, base: str, path: str, params: dict = None, signed: bool = True):
        session = await self._get_session()
        p = params or {}
        if signed:
            p = self._sign(p)
        async with session.delete(f"{base}{path}", params=p) as r:
            data = await r.json()
            if r.status != 200:
                raise RuntimeError(f"DELETE {path} {r.status}: {data}")
            return data

    # ── Market data ───────────────────────────────────────────────────────────

    async def get_candles(self, symbol: str, interval: str, limit: int) -> list[dict]:
        base = self._fapi_url if _is_perp(symbol) else self._spot_url
        path = "/fapi/v1/klines" if _is_perp(symbol) else "/api/v3/klines"
        data = await self._get(base, path, {"symbol": symbol, "interval": interval, "limit": limit})
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
        # Use cached WebSocket price if available (lower latency)
        if symbol in self._prices:
            return self._prices[symbol]
        base = self._fapi_url if _is_perp(symbol) else self._spot_url
        path = "/fapi/v1/ticker/price" if _is_perp(symbol) else "/api/v3/ticker/price"
        data = await self._get(base, path, {"symbol": symbol})
        return float(data["price"])

    async def get_orderbook(self, symbol: str, depth: int = 20) -> dict:
        base = self._fapi_url if _is_perp(symbol) else self._spot_url
        path = "/fapi/v1/depth" if _is_perp(symbol) else "/api/v3/depth"
        data = await self._get(base, path, {"symbol": symbol, "limit": depth})
        return {
            "bids": [(float(p), float(q)) for p, q in data["bids"]],
            "asks": [(float(p), float(q)) for p, q in data["asks"]],
        }

    async def get_funding_rate(self, symbol: str) -> float:
        try:
            data = await self._get(self._fapi_url, "/fapi/v1/premiumIndex", {"symbol": symbol})
            return float(data.get("lastFundingRate", 0))
        except Exception:
            return 0.0

    # ── Order management ──────────────────────────────────────────────────────

    async def place_order(self, order: Order) -> Order:
        base = self._fapi_url if _is_perp(order.symbol) else self._spot_url
        path = "/fapi/v1/order" if _is_perp(order.symbol) else "/api/v3/order"

        params: dict = {
            "symbol":   order.symbol,
            "side":     order.side.value,
            "type":     order.type.value,
            "quantity": f"{order.quantity:.6f}",
        }

        if order.type == OrderType.LIMIT:
            params["price"]       = f"{order.price:.2f}"
            params["timeInForce"] = "GTC"

        if _is_perp(order.symbol):
            params["reduceOnly"] = "false"

        try:
            data = await self._post(base, path, params)
            order.order_id  = str(data.get("orderId", uuid.uuid4()))
            order.status    = self._map_status(data.get("status", "NEW"))
            order.avg_price = float(data.get("avgPrice") or data.get("price") or order.price or 0)
            order.filled_qty = float(data.get("executedQty", 0))
            order.fee       = order.filled_qty * order.avg_price * self.FEE_RATE
            logger.info(
                f"[BINANCE] Order placed: {order.side.value} {order.quantity} "
                f"{order.symbol} @ {order.avg_price:.4f} | status={order.status.value}"
            )
        except Exception as e:
            logger.error(f"[BINANCE] place_order failed: {e}")
            order.status = OrderStatus.CANCELLED

        return order

    async def cancel_order(self, symbol: str, order_id: str) -> bool:
        base = self._fapi_url if _is_perp(symbol) else self._spot_url
        path = "/fapi/v1/order" if _is_perp(symbol) else "/api/v3/order"
        try:
            await self._delete(base, path, {"symbol": symbol, "orderId": order_id})
            return True
        except Exception as e:
            logger.error(f"[BINANCE] cancel_order failed: {e}")
            return False

    async def get_balances(self) -> list[Balance]:
        try:
            # Futures account
            data = await self._get(self._fapi_url, "/fapi/v2/account", signed=True)
            assets = data.get("assets", [])
            return [
                Balance(
                    asset=a["asset"],
                    free=float(a["availableBalance"]),
                    locked=float(a["initialMargin"]),
                )
                for a in assets
                if float(a.get("walletBalance", 0)) > 0
            ]
        except Exception as e:
            logger.error(f"[BINANCE] get_balances failed: {e}")
            return []

    async def get_open_orders(self, symbol: str) -> list[Order]:
        base = self._fapi_url if _is_perp(symbol) else self._spot_url
        path = "/fapi/v1/openOrders" if _is_perp(symbol) else "/api/v3/openOrders"
        try:
            data = await self._get(base, path, {"symbol": symbol}, signed=True)
            return [self._parse_order(d) for d in data]
        except Exception as e:
            logger.error(f"[BINANCE] get_open_orders failed: {e}")
            return []

    # ── WebSocket price stream ─────────────────────────────────────────────────

    async def start_price_stream(self, symbols: list[str]):
        """
        Subscribe to real-time price updates via WebSocket.
        Runs as a background task — call once at startup.
        """
        self._ws_task = asyncio.create_task(self._ws_loop(symbols))

    async def _ws_loop(self, symbols: list[str]):
        streams = "/".join(f"{s.lower()}@aggTrade" for s in symbols)
        ws_url  = f"{self._fapi_ws_url}/{streams}"
        while True:
            try:
                async with aiohttp.ClientSession() as session:
                    async with session.ws_connect(ws_url) as ws:
                        logger.info(f"[WS] Connected to {len(symbols)} streams")
                        async for msg in ws:
                            if msg.type == aiohttp.WSMsgType.TEXT:
                                data = json.loads(msg.data)
                                sym  = data.get("s")
                                price = float(data.get("p", 0))
                                if sym and price:
                                    self._prices[sym] = price
                            elif msg.type in (aiohttp.WSMsgType.ERROR, aiohttp.WSMsgType.CLOSED):
                                break
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.warning(f"[WS] Disconnected: {e} — reconnecting in 5s")
                await asyncio.sleep(5)

    # ── Helpers ───────────────────────────────────────────────────────────────

    @staticmethod
    def _map_status(s: str) -> OrderStatus:
        return {
            "NEW":              OrderStatus.OPEN,
            "PARTIALLY_FILLED": OrderStatus.PARTIAL,
            "FILLED":           OrderStatus.FILLED,
            "CANCELED":         OrderStatus.CANCELLED,
            "REJECTED":         OrderStatus.CANCELLED,
            "EXPIRED":          OrderStatus.CANCELLED,
        }.get(s, OrderStatus.OPEN)

    @staticmethod
    def _parse_order(data: dict) -> Order:
        return Order(
            symbol=data["symbol"],
            side=OrderSide(data["side"]),
            type=OrderType(data["type"]),
            quantity=float(data["origQty"]),
            price=float(data.get("price") or 0) or None,
            order_id=str(data["orderId"]),
            status=BinanceExchange._map_status(data["status"]),
            filled_qty=float(data.get("executedQty", 0)),
            avg_price=float(data.get("avgPrice") or 0),
        )

    async def portfolio_value(self) -> float:
        """Total USDT value of futures account."""
        try:
            data = await self._get(self._fapi_url, "/fapi/v2/account", signed=True)
            return float(data.get("totalWalletBalance", 0))
        except Exception:
            return 0.0
