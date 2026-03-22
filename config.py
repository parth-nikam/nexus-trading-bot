# nexus/config.py — Central configuration with startup validation
import os
from dotenv import load_dotenv

load_dotenv()

# ── Exchange ──────────────────────────────────────────────────────────────────
BINANCE_API_KEY    = os.getenv("BINANCE_API_KEY", "")
BINANCE_API_SECRET = os.getenv("BINANCE_API_SECRET", "")
BINANCE_TESTNET    = os.getenv("BINANCE_TESTNET", "true").lower() == "true"

# ── Trading universe ──────────────────────────────────────────────────────────
SYMBOLS         = ["BTCUSDT", "ETHUSDT"]
CANDLE_INTERVAL = "5m"
CANDLE_LIMIT    = 200

# ── Execution ─────────────────────────────────────────────────────────────────
PAPER_TRADING   = os.getenv("PAPER_TRADING", "true").lower() != "false"
INITIAL_CAPITAL = float(os.getenv("INITIAL_CAPITAL", "10000"))
TRADE_SIZE_PCT  = 0.10      # base % of capital per trade (Kelly-adjusted)
MAX_POSITIONS   = 3
SLIPPAGE_BPS    = 5

# ── Risk ──────────────────────────────────────────────────────────────────────
STOP_LOSS_PCT      = 1.5
TAKE_PROFIT_PCT    = 3.0
MAX_DAILY_LOSS_PCT = 5.0
MAX_DRAWDOWN_PCT   = 15.0

# ── Strategy voting ───────────────────────────────────────────────────────────
BUY_THRESHOLD  = 0.35
SELL_THRESHOLD = 0.35

# ── Circuit breaker ───────────────────────────────────────────────────────────
CIRCUIT_BREAKER_ERRORS   = 5    # consecutive errors before halting
CIRCUIT_BREAKER_COOLDOWN = 300  # seconds to wait before resuming (5 min)

# ── Order management ──────────────────────────────────────────────────────────
LIMIT_ORDER_TIMEOUT = 300   # cancel unfilled limit orders after 5 min

# ── Volatility scaling ────────────────────────────────────────────────────────
VOL_LOOKBACK    = 20        # candles for realized vol calculation
VOL_SCALE_HIGH  = 0.03      # daily vol > 3% → scale position down
VOL_SCALE_LOW   = 0.01      # daily vol < 1% → scale position up

# ── Infrastructure ────────────────────────────────────────────────────────────
DASHBOARD_PORT = 8080
LOG_LEVEL      = "INFO"


def validate():
    """Called at startup — raises ValueError on bad config."""
    assert 0 < TRADE_SIZE_PCT <= 1.0,   f"TRADE_SIZE_PCT must be 0-1, got {TRADE_SIZE_PCT}"
    assert STOP_LOSS_PCT > 0,           f"STOP_LOSS_PCT must be > 0"
    assert TAKE_PROFIT_PCT > 0,         f"TAKE_PROFIT_PCT must be > 0"
    assert TAKE_PROFIT_PCT > STOP_LOSS_PCT, "TAKE_PROFIT_PCT must exceed STOP_LOSS_PCT"
    assert MAX_POSITIONS >= 1,          f"MAX_POSITIONS must be >= 1"
    assert INITIAL_CAPITAL > 0,         f"INITIAL_CAPITAL must be > 0"
    assert 0 < BUY_THRESHOLD <= 1.0,    f"BUY_THRESHOLD must be 0-1"
    assert len(SYMBOLS) > 0,            "SYMBOLS must not be empty"
    if not PAPER_TRADING:
        assert BINANCE_API_KEY,    "BINANCE_API_KEY required for live trading"
        assert BINANCE_API_SECRET, "BINANCE_API_SECRET required for live trading"
