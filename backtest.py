"""
backtest.py — Nexus backtester (v3)

Uses 1h candles, ATR-based SL/TP, 2-bar no-stop zone, trailing stop.
Sharpe computed from daily equity returns (not per-trade).

Usage:
    python backtest.py --symbol BTCUSDT --days 90
    python backtest.py --symbol ETHUSDT --days 30
"""

from __future__ import annotations

import argparse
import asyncio
import math
import time
from typing import Optional

import aiohttp
import numpy as np
import pandas as pd

import config
from alpha.engine import AlphaEngine
from utils.logger import get_logger

logger = get_logger("nexus.backtest")

BINANCE_PUBLIC = "https://api.binance.com"

# ── Constants (mirrors cmd/backtest/main.go) ──────────────────────────────────
WARMUP        = 215   # bars needed for EMA200 + indicators
BARS_PER_DAY  = 24    # 1h candles
FEE_RATE      = 0.0008
MIN_HOLD_BARS = 3
MAX_HOLD_BARS = 72
NO_STOP_BARS  = 2     # no SL check for first 2 bars after entry

# ATR-based SL/TP with percentage caps
SL_ATR_MULT = 2.0
TP_ATR_MULT = 3.5
SL_MIN_PCT  = 0.012
SL_MAX_PCT  = 0.040
TP_MIN_PCT  = 0.025
TP_MAX_PCT  = 0.090
TRAIL_PCT   = 0.030


# ── Data fetching ─────────────────────────────────────────────────────────────

async def fetch_candles(
    symbol: str,
    days: int,
    session: aiohttp.ClientSession,
    interval: str = "1h",
) -> pd.DataFrame:
    needed    = days * BARS_PER_DAY + WARMUP
    end_ms    = int(time.time() * 1000)
    all_rows: list = []

    logger.info(f"Fetching {needed} {interval} candles for {symbol}...")

    while len(all_rows) < needed:
        batch = min(1000, needed - len(all_rows))
        params: dict = {"symbol": symbol, "interval": interval, "limit": batch}
        if all_rows:
            params["endTime"] = all_rows[0][0] - 1  # walk backwards

        async with session.get(
            f"{BINANCE_PUBLIC}/api/v3/klines",
            params=params,
            timeout=aiohttp.ClientTimeout(total=15),
        ) as r:
            r.raise_for_status()
            rows = await r.json()

        if not rows:
            break
        all_rows = rows + all_rows
        if len(rows) < batch:
            break

    # Trim to needed
    if len(all_rows) > needed:
        all_rows = all_rows[-needed:]

    df = pd.DataFrame(all_rows, columns=[
        "time", "open", "high", "low", "close", "volume",
        "close_time", "qav", "trades", "tbbav", "tbqav", "ignore",
    ])
    for col in ("open", "high", "low", "close", "volume"):
        df[col] = df[col].astype(float)
    df["time"] = pd.to_datetime(df["time"], unit="ms")
    df = df.set_index("time").sort_index().reset_index(drop=True)
    logger.info(f"Fetched {len(df)} candles")
    return df


# ── ATR helper ────────────────────────────────────────────────────────────────

def compute_atr(df: pd.DataFrame, i: int, period: int = 14) -> float:
    """Wilder ATR at bar i."""
    if i < period:
        return 0.0
    hi = df["high"].values
    lo = df["low"].values
    cl = df["close"].values
    trs = []
    for j in range(1, i + 1):
        tr = max(hi[j] - lo[j], abs(hi[j] - cl[j - 1]), abs(lo[j] - cl[j - 1]))
        trs.append(tr)
    # Wilder smoothing
    atr = sum(trs[:period]) / period
    for tr in trs[period:]:
        atr = (atr * (period - 1) + tr) / period
    return atr


# ── Core backtest ─────────────────────────────────────────────────────────────

def run_backtest(df: pd.DataFrame, capital: float = 10_000.0) -> dict:
    if len(df) < WARMUP + 10:
        return {"error": f"not enough candles: {len(df)}"}

    engine = AlphaEngine()

    # Pre-compute signals
    signals: list[Optional[tuple]] = [None] * len(df)
    dist = {"BUY": 0, "SELL": 0, "HOLD": 0}
    for i in range(WARMUP, len(df)):
        window = df.iloc[: i + 1].copy()
        sig = engine.evaluate("BT", window)
        signals[i] = (sig.action, sig.score, sig.agreeing)
        dist[sig.action] += 1

    logger.info(f"Signal dist — BUY={dist['BUY']} SELL={dist['SELL']} HOLD={dist['HOLD']}")

    peak    = capital
    max_dd  = 0.0
    lev     = float(getattr(config, "LEVERAGE", 3))
    size_pct = getattr(config, "TRADE_SIZE_PCT", 0.20)

    trades: list[dict] = []
    equity_series = [capital] * len(df)

    # Position state
    in_trade       = False
    open_price     = 0.0
    open_notional  = 0.0
    open_side      = ""
    open_bar       = 0
    open_sl        = 0.0
    open_tp        = 0.0
    open_atr       = 0.0
    trail_active   = False
    breakeven_active = False
    peak_price     = 0.0
    trail_stop     = 0.0

    def close_trade(exit_price: float, bar: int, reason: str):
        nonlocal capital, peak, max_dd, in_trade
        nonlocal trail_active, breakeven_active, peak_price, trail_stop

        fee = open_notional * FEE_RATE * 2
        if open_side == "BUY":
            pnl = (exit_price - open_price) / open_price * open_notional - fee
        else:
            pnl = (open_price - exit_price) / open_price * open_notional - fee

        capital += pnl
        hold = bar - open_bar
        trades.append({
            "side": open_side, "entry": open_price, "exit": exit_price,
            "pnl": round(pnl, 4), "hold": hold, "reason": reason,
        })
        if capital > peak:
            peak = capital
        dd = (peak - capital) / peak * 100
        if dd > max_dd:
            max_dd = dd

        logger.info(
            f"CLOSE {open_side} bar={bar} entry={open_price:.2f} exit={exit_price:.2f} "
            f"pnl={pnl:+.2f} hold={hold} [{reason}] cap={capital:.2f}"
        )
        in_trade = trail_active = breakeven_active = False
        peak_price = trail_stop = 0.0

    for i in range(WARMUP, len(df)):
        c = df.iloc[i]
        equity_series[i] = capital

        if in_trade:
            # Update trailing stop
            if open_side == "BUY":
                if c["high"] > peak_price:
                    peak_price = c["high"]
                profit_pct = (peak_price - open_price) / open_price
                if not breakeven_active and open_atr > 0 and (c["high"] - open_price) >= open_atr * 3.0:
                    breakeven_active = True
                    be_level = open_price * 1.001
                    if open_sl < be_level:
                        open_sl = be_level
                if profit_pct >= TRAIL_PCT:
                    trail_active = True
                    new_trail = peak_price * (1 - TRAIL_PCT)
                    if new_trail > trail_stop:
                        trail_stop = new_trail
            else:
                if peak_price == 0 or c["low"] < peak_price:
                    peak_price = c["low"]
                profit_pct = (open_price - peak_price) / open_price
                if not breakeven_active and open_atr > 0 and (open_price - c["low"]) >= open_atr * 3.0:
                    breakeven_active = True
                    be_level = open_price * 0.999
                    if open_sl > be_level:
                        open_sl = be_level
                if profit_pct >= TRAIL_PCT:
                    trail_active = True
                    new_trail = peak_price * (1 + TRAIL_PCT)
                    if trail_stop == 0 or new_trail < trail_stop:
                        trail_stop = new_trail

            sl = trail_stop if trail_active else open_sl
            no_stop_zone = (i - open_bar) < NO_STOP_BARS

            if open_side == "BUY":
                if not no_stop_zone and c["low"] <= sl:
                    exit_p = sl if c["open"] >= sl else c["open"]
                    close_trade(exit_p, i, "stop_loss")
                    continue
                if c["high"] >= open_tp:
                    close_trade(open_tp, i, "take_profit")
                    continue
            else:
                if not no_stop_zone and c["high"] >= sl:
                    exit_p = sl if c["open"] <= sl else c["open"]
                    close_trade(exit_p, i, "stop_loss")
                    continue
                if c["low"] <= open_tp:
                    close_trade(open_tp, i, "take_profit")
                    continue

            if i - open_bar >= MAX_HOLD_BARS:
                close_trade(c["close"], i, "max_hold")
                continue

            if i - open_bar >= MIN_HOLD_BARS and signals[i]:
                action, score, _ = signals[i]
                if (open_side == "BUY" and action == "SELL" and score > 0.35) or \
                   (open_side == "SELL" and action == "BUY" and score > 0.35):
                    close_trade(c["close"], i, "signal_flip")
                    continue

        if not in_trade and signals[i]:
            action, score, _ = signals[i]
            if action in ("BUY", "SELL"):
                margin = capital * size_pct
                if margin <= 0:
                    continue

                open_notional = margin * lev
                open_price    = c["close"]
                open_side     = action

                atr = compute_atr(df, i, 14)
                if atr == 0:
                    atr = open_price * 0.01

                sl_dist = max(open_price * SL_MIN_PCT, min(open_price * SL_MAX_PCT, atr * SL_ATR_MULT))
                tp_dist = max(open_price * TP_MIN_PCT, min(open_price * TP_MAX_PCT, atr * TP_ATR_MULT))

                if action == "BUY":
                    open_sl = open_price - sl_dist
                    open_tp = open_price + tp_dist
                else:
                    open_sl = open_price + sl_dist
                    open_tp = open_price - tp_dist

                open_bar       = i
                open_atr       = atr
                in_trade       = True
                peak_price     = open_price
                trail_active   = False
                breakeven_active = False
                trail_stop     = 0.0
                capital       -= open_notional * FEE_RATE

                sl_pct = abs(open_sl - open_price) / open_price * 100
                tp_pct = abs(open_tp - open_price) / open_price * 100
                logger.info(
                    f"OPEN {action} bar={i} price={open_price:.2f} notional={open_notional:.2f} "
                    f"score={score:.3f} SL={open_sl:.2f}({sl_pct:.2f}%) TP={open_tp:.2f}({tp_pct:.2f}%) cap={capital:.2f}"
                )

    if in_trade:
        close_trade(df.iloc[-1]["close"], len(df) - 1, "end_of_data")

    return _compute_stats(trades, equity_series, capital, 10_000.0, max_dd, dist)


def _compute_stats(
    trades: list[dict],
    equity_series: list[float],
    final_capital: float,
    initial_capital: float,
    max_dd: float,
    dist: dict,
) -> dict:
    n = len(trades)
    if n == 0:
        return {
            "return_pct": 0.0, "trade_count": 0, "win_rate": 0.0,
            "sharpe": 0.0, "max_drawdown": 0.0, "profit_factor": 0.0,
            "avg_hold_bars": 0.0, "final_capital": round(final_capital, 2),
            "gross_wins": 0.0, "gross_losses": 0.0, "signal_dist": dist,
        }

    wins = [t for t in trades if t["pnl"] > 0]
    losses = [t for t in trades if t["pnl"] <= 0]
    gross_w = sum(t["pnl"] for t in wins)
    gross_l = sum(abs(t["pnl"]) for t in losses)
    total_hold = sum(t["hold"] for t in trades)

    wr = len(wins) / n
    avg_hold = total_hold / n
    pf = gross_w / gross_l if gross_l > 0 else (99.0 if gross_w > 0 else 0.0)
    ret_pct = (final_capital - initial_capital) / initial_capital * 100

    # Sharpe: daily equity returns (24 bars/day for 1h candles)
    sharpe = 0.0
    trading_bars = len(equity_series) - WARMUP
    if trading_bars >= BARS_PER_DAY * 2:
        num_days = trading_bars // BARS_PER_DAY
        daily_rets = []
        for d in range(num_days):
            s = WARMUP + d * BARS_PER_DAY
            e = min(s + BARS_PER_DAY - 1, len(equity_series) - 1)
            if equity_series[s] > 0:
                daily_rets.append((equity_series[e] - equity_series[s]) / equity_series[s])
        if len(daily_rets) >= 2:
            mean = sum(daily_rets) / len(daily_rets)
            variance = sum((r - mean) ** 2 for r in daily_rets) / len(daily_rets)
            std = math.sqrt(variance)
            if std > 0:
                sharpe = mean / std * math.sqrt(252)

    exit_reasons: dict[str, int] = {}
    for t in trades:
        exit_reasons[t["reason"]] = exit_reasons.get(t["reason"], 0) + 1

    logger.info(
        f"DONE trades={n} wins={len(wins)} wr={wr*100:.1f}% "
        f"return={ret_pct:.2f}% PF={pf:.2f} sharpe={sharpe:.2f} maxDD={max_dd:.2f}%"
    )

    return {
        "return_pct":    round(ret_pct, 2),
        "trade_count":   n,
        "win_rate":      round(wr, 3),
        "sharpe":        round(sharpe, 2),
        "max_drawdown":  round(max_dd, 2),
        "profit_factor": round(pf, 2),
        "avg_hold_bars": round(avg_hold, 1),
        "final_capital": round(final_capital, 2),
        "gross_wins":    round(gross_w, 2),
        "gross_losses":  round(gross_l, 2),
        "signal_dist":   dist,
        "exit_reasons":  exit_reasons,
    }


# ── Dashboard-callable wrapper ────────────────────────────────────────────────

async def run_backtest_async(symbol: str, days: int) -> dict:
    async with aiohttp.ClientSession() as session:
        df = await fetch_candles(symbol, days, session)
    return run_backtest(df)


# ── CLI ───────────────────────────────────────────────────────────────────────

async def main():
    parser = argparse.ArgumentParser(description="Nexus Backtester")
    parser.add_argument("--symbol", default="BTCUSDT")
    parser.add_argument("--days",   type=int, default=90)
    args = parser.parse_args()

    async with aiohttp.ClientSession() as session:
        df = await fetch_candles(args.symbol, args.days, session)

    stats = run_backtest(df)
    print(f"\n{'='*50}")
    print(f"  NEXUS BACKTEST — {args.symbol} {args.days}d (1h candles)")
    print(f"{'='*50}")
    for k, v in stats.items():
        print(f"  {k:<20} {v}")
    print(f"{'='*50}\n")


if __name__ == "__main__":
    asyncio.run(main())
