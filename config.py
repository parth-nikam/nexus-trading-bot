# nexus/config.py — Central configuration
# Exchange: Binance (USDT perpetuals + spot)
# All secrets loaded from .env

import os
from dotenv import load_dotenv

load_dotenv()

# ── Exchange ──────────────────────────────────────────────────────────────────
BINANCE_API_KEY    = os.getenv("BINANCE_API_KEY", "")
BINANCE_API_SECRET = os.getenv("BINANCE_API_SECRET", "")
BINANCE_TESTNET    = os.getenv("BINANCE_TESTNET", "true").lower() == "true"

# ── Trading universe ──────────────────────────────────────────────────────────
# Start with BTC and ETH perps — deepest liquidity, tightest spreads
SYMBOLS = ["BTCUSDT", "ETHUSDT"]
CANDLE_INTERVAL = "5m"     # 5m for intraday alpha
CANDLE_LIMIT    = 200      # warmup candles

# ── Execution ─────────────────────────────────────────────────────────────────
PAPER_TRADING   = True     # flip to False for live
INITIAL_CAPITAL = 10_000   # USD — paper trading starting balance
TRADE_SIZE_PCT  = 0.10     # 10% of capital per trade (Kelly-adjusted)
MAX_POSITIONS   = 3        # max concurrent open positions
SLIPPAGE_BPS    = 5        # assumed slippage in basis points

# ── Risk ──────────────────────────────────────────────────────────────────────
STOP_LOSS_PCT      = 1.5   # % from entry
TAKE_PROFIT_PCT    = 3.0   # % from entry
MAX_DAILY_LOSS_PCT = 5.0   # halt trading if daily drawdown > 5%
MAX_DRAWDOWN_PCT   = 15.0  # halt if portfolio drawdown > 15%

# ── Strategy voting ───────────────────────────────────────────────────────────
BUY_THRESHOLD  = 0.50      # weighted score to fire BUY
SELL_THRESHOLD = 0.50

# ── Infrastructure ────────────────────────────────────────────────────────────
REDIS_URL  = os.getenv("REDIS_URL", "redis://localhost:6379")
DB_URL     = os.getenv("DATABASE_URL", "postgresql://localhost/nexus")
DASHBOARD_PORT = 8080
LOG_LEVEL  = "INFO"
