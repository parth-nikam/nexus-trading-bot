"""
dashboard/app.py — FastAPI dashboard for Nexus Trading System.

REST:
  GET  /                        — HTML dashboard
  GET  /api/status              — portfolio snapshot
  GET  /api/trades              — trade history
  GET  /api/positions           — open positions
  GET  /api/candles/{symbol}    — OHLCV for chart
  GET  /api/equity              — equity curve history
  GET  /api/strategy_perf       — per-strategy metrics
  GET  /api/circuit_breakers    — circuit breaker status
  POST /api/control             — force BUY/SELL/HALT/RESUME

WebSocket:
  WS /ws — pushes every second: prices, pnl, votes, equity point, alerts
"""

from __future__ import annotations

import asyncio
import json
import time
from pathlib import Path
from typing import TYPE_CHECKING

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse

if TYPE_CHECKING:
    from bot import NexusBot

app = FastAPI(title="Nexus", version="2.0.0")

_bot: "NexusBot | None" = None
_ws_clients: list[WebSocket] = []
_equity_history: list[dict] = []
_last_equity_ts: float = 0
_alerts: list[dict] = []   # recent alerts for dashboard


def attach_bot(bot: "NexusBot"):
    global _bot
    _bot = bot


def _add_alert(level: str, msg: str):
    _alerts.append({"time": int(time.time()), "level": level, "msg": msg})
    if len(_alerts) > 50:
        _alerts.pop(0)


# ── REST ──────────────────────────────────────────────────────────────────────

@app.get("/", response_class=HTMLResponse)
async def index():
    return (Path(__file__).parent / "templates" / "index.html").read_text()


@app.get("/api/status")
async def status():
    if not _bot:
        return {"error": "bot not running"}
    summary = _bot._risk.summary()
    pv = await _bot._exchange.portfolio_value()
    return {
        "portfolio_value": round(pv, 2),
        **summary,
        "cycle":   _bot._cycle,
        "running": _bot._running,
        "mode":    "PAPER" if __import__("config").PAPER_TRADING else "LIVE",
    }


@app.get("/api/trades")
async def trades():
    if not _bot:
        return []
    return _bot._exchange.trade_history()


@app.get("/api/positions")
async def positions():
    if not _bot:
        return {}
    return {
        sym: {
            "side":           pos.side,
            "quantity":       pos.quantity,
            "entry_price":    pos.entry_price,
            "current_price":  pos.current_price,
            "unrealized_pnl": pos.unrealized_pnl,
            "strategy":       pos.strategy,
        }
        for sym, pos in _bot._risk.open_positions.items()
    }


@app.get("/api/candles/{symbol}")
async def candles(symbol: str, interval: str = "5m", limit: int = 150):
    if not _bot:
        return []
    return await _bot._exchange.get_candles(symbol.upper(), interval, limit)


@app.get("/api/equity")
async def equity():
    return _equity_history[-500:]


@app.get("/api/strategy_perf")
async def strategy_perf():
    if not _bot:
        return {}
    return _bot._risk.strategy_performance()


@app.get("/api/circuit_breakers")
async def circuit_breakers():
    if not _bot:
        return []
    return _bot.circuit_breaker_status()


@app.get("/api/alerts")
async def alerts():
    return list(reversed(_alerts[-20:]))


@app.post("/api/control")
async def control(body: dict):
    if not _bot:
        return {"error": "bot not running"}
    action = body.get("action", "").lower()
    symbol = body.get("symbol", "BTCUSDT").upper()

    if action == "halt":
        _bot.stop()
        _bot._risk.halt("manual_dashboard")
        _add_alert("warn", "Bot halted via dashboard")
        return {"ok": True, "action": "halt"}

    if action == "resume":
        _bot._risk.resume()
        _add_alert("info", "Bot resumed via dashboard")
        return {"ok": True, "action": "resume"}

    if action == "reset_cb":
        for cb in _bot._cb.values():
            cb.reset()
        _add_alert("info", "Circuit breakers reset")
        return {"ok": True, "action": "reset_cb"}

    if action in ("buy", "sell"):
        # Cancel any pending limit orders for this symbol first
        open_orders = await _bot._exchange.get_open_orders(symbol)
        for o in open_orders:
            await _bot._exchange.cancel_order(symbol, o.order_id)

        from alpha.engine import Signal
        sig = Signal(
            symbol=symbol, action=action.upper(),
            score=1.0, kelly_f=0.05, breakdown={}, reason="manual_override",
        )
        asyncio.create_task(_bot._router.process_market(sig))
        _add_alert("info", f"Force {action.upper()} {symbol}")
        return {"ok": True, "action": action, "symbol": symbol}

    return {"error": f"unknown action: {action}"}


# ── WebSocket ─────────────────────────────────────────────────────────────────

@app.websocket("/ws")
async def ws_endpoint(ws: WebSocket):
    await ws.accept()
    _ws_clients.append(ws)
    try:
        while True:
            await asyncio.sleep(1)
            if not _bot:
                continue

            summary = _bot._risk.summary()
            pv = await _bot._exchange.portfolio_value()

            # Equity history
            global _last_equity_ts
            now = int(time.time())
            if now - _last_equity_ts >= 5:
                _equity_history.append({"time": now, "value": round(pv, 2)})
                _last_equity_ts = now
                # Prune to last 2000 points
                if len(_equity_history) > 2000:
                    _equity_history.pop(0)

            # Live prices
            prices = {}
            for sym in config_symbols():
                try:
                    prices[sym] = await _bot._exchange.get_ticker(sym)
                except Exception:
                    pass

            # Strategy votes
            votes = {}
            try:
                last_votes = getattr(_bot._alpha, "_last_votes", {})
                for sym_votes in last_votes.values():
                    votes = sym_votes
                    break
            except Exception:
                pass

            # Circuit breaker status
            cb_status = _bot.circuit_breaker_status()
            any_open  = any(c["state"] == "OPEN" for c in cb_status)

            # Alert on drawdown
            if summary["drawdown_pct"] > 5:
                _add_alert("warn", f"Drawdown {summary['drawdown_pct']:.1f}%")

            await ws.send_text(json.dumps({
                "portfolio_value": round(pv, 2),
                "total_pnl":       summary["total_pnl"],
                "win_rate":        summary["win_rate"],
                "drawdown_pct":    summary["drawdown_pct"],
                "trade_count":     summary["trade_count"],
                "open_count":      summary["open_count"],
                "avg_hold_min":    summary["avg_hold_min"],
                "cycle":           _bot._cycle,
                "halted":          summary["halted"],
                "halt_reason":     summary["halt_reason"],
                "prices":          prices,
                "equity_point":    {"time": now, "value": round(pv, 2)},
                "votes":           votes,
                "cb_open":         any_open,
                "alerts":          _alerts[-3:],
            }))
    except WebSocketDisconnect:
        pass
    except Exception:
        pass
    finally:
        if ws in _ws_clients:
            _ws_clients.remove(ws)


def config_symbols():
    try:
        import config
        return config.SYMBOLS
    except Exception:
        return ["BTCUSDT", "ETHUSDT"]
