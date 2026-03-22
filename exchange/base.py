# exchange/base.py — Abstract exchange interface
# Every exchange (Binance, OKX, paper) implements this contract.
# The execution layer never touches exchange-specific code directly.

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Optional
from enum import Enum


class OrderSide(str, Enum):
    BUY  = "BUY"
    SELL = "SELL"

class OrderType(str, Enum):
    LIMIT  = "LIMIT"
    MARKET = "MARKET"

class OrderStatus(str, Enum):
    OPEN      = "OPEN"
    FILLED    = "FILLED"
    CANCELLED = "CANCELLED"
    PARTIAL   = "PARTIAL"


@dataclass
class Order:
    symbol:     str
    side:       OrderSide
    type:       OrderType
    quantity:   float
    price:      Optional[float] = None   # None for market orders
    order_id:   Optional[str]  = None
    status:     OrderStatus    = OrderStatus.OPEN
    filled_qty: float          = 0.0
    avg_price:  float          = 0.0
    fee:        float          = 0.0
    timestamp:  Optional[float] = None


@dataclass
class Position:
    symbol:         str
    side:           OrderSide
    quantity:       float
    entry_price:    float
    current_price:  float = 0.0
    unrealized_pnl: float = 0.0
    realized_pnl:   float = 0.0

    def update_price(self, price: float):
        self.current_price = price
        mult = 1 if self.side == OrderSide.BUY else -1
        self.unrealized_pnl = mult * (price - self.entry_price) * self.quantity


@dataclass
class Balance:
    asset:     str
    free:      float
    locked:    float
    total:     float = field(init=False)

    def __post_init__(self):
        self.total = self.free + self.locked


class AbstractExchange(ABC):

    @abstractmethod
    async def get_candles(self, symbol: str, interval: str, limit: int) -> list[dict]:
        ...

    @abstractmethod
    async def get_ticker(self, symbol: str) -> float:
        ...

    @abstractmethod
    async def get_orderbook(self, symbol: str, depth: int = 20) -> dict:
        ...

    @abstractmethod
    async def place_order(self, order: Order) -> Order:
        ...

    @abstractmethod
    async def cancel_order(self, symbol: str, order_id: str) -> bool:
        ...

    @abstractmethod
    async def get_balances(self) -> list[Balance]:
        ...

    @abstractmethod
    async def get_open_orders(self, symbol: str) -> list[Order]:
        ...
