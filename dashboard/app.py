"""
dashboard/app.py — FastAPI dashboard for Nexus Trading System.

Endpoints:
  GET  /              — HTML dashboard
  GET  /api/status    — portfolio snapshot (JSON)
  GET  /api/trades    — trade history (JSON)
  GET  /api/positions — open positions (JSON)
  POST /api/control   — force BUY/SELL/HALT (JSON body: {"action": "buy", "symbol": "BTCUSDT"})
  WS   /ws            — real-time updates pushed every second
"""

from __future__ import annotations

import asyncio
import json
from typing import TYPE_CHECKING

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse

if TYPE_CHECKING:
    from bot import NexusBot

app = FastAPI(title="Nexus Trading System", version="1.0.0")

# Shared bot reference — injected at startup
_bot: "NexusBot | None" = None
_ws_clients: list[WebSocket] = []


def attach_bot(bot: "NexusBot"):
    global _bot
    _bot = bot


# ── REST endpoints ────────────────────────────────────────────────────────────

@app.get("/", response_class=HTMLResponse)
async def index():
    with open("dashboard/templates/index.html") as f:
        return f.read()


@app.get("/api/status")
async def status():
    if not _bot:
        return {"error": "bot not running"}
    summary = _bot._risk.summary()
    pv = await _bot._exchange.portfolio_value()
    return {
        "portfolio_value": round(pv, 2),
        "capital":         summary["capital"],
        "total_pnl":       summary["total_pnl"],
        "trade_count":     summary["trade_count"],
        "win_rate":        summary["win_rate"],
        "open_count":      summary["open_count"],
        "drawdown_pct":    summary["drawdown_pct"],
        "cycle":           _bot._cycle,
        "running":         _bot._running,
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
            "side":          pos.side,
            "quantity":      pos.quantity,
            "entry_price":   pos.entry_price,
            "current_price": pos.current_price,
            "unrealized_pnl": pos.unrealized_pnl,
        }
        for sym, pos in _bot._risk.open_positions.items()
    }


@app.post("/api/control")
async def control(body: dict):
    if not _bot:
        return {"error": "bot not running"}
    action = body.get("action", "").lower()
    symbol = body.get("symbol", "BTCUSDT").upper()

    if action == "halt":
        _bot.stop()
        return {"ok": True, "action": "halt"}

    if action in ("buy", "sell"):
        from alpha.engine import Signal
        sig = Signal(
            symbol=symbol,
            action=action.upper(),
            score=1.0,
            kelly_f=0.05,
            breakdown={},
            reason="manual_override",
        )
        asyncio.create_task(_bot._router.process(sig))
        return {"ok": True, "action": action, "symbol": symbol}

    return {"error": f"unknown action: {action}"}


# ── WebSocket push ────────────────────────────────────────────────────────────

@app.websocket("/ws")
async def websocket_endpoint(ws: WebSocket):
    await ws.accept()
    _ws_clients.append(ws)
    try:
        while True:
            await asyncio.sleep(1)
            if _bot:
                summary = _bot._risk.summary()
                pv = await _bot._exchange.portfolio_value()
                await ws.send_text(json.dumps({
                    "portfolio_value": round(pv, 2),
                    "total_pnl":       summary["total_pnl"],
                    "win_rate":        summary["win_rate"],
                    "drawdown_pct":    summary["drawdown_pct"],
                    "cycle":           _bot._cycle,
                }))
    except WebSocketDisconnect:
        _ws_clients.remove(ws)
    except Exception:
        if ws in _ws_clients:
            _ws_clients.remove(ws)
