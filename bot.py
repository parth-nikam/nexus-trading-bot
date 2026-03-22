#!/usr/bin/env python3
"""
bot.py — Nexus Trading System entry point.

Runs two concurrent async tasks:
  1. Strategy loop  — fetches candles + OB, runs alpha engine, routes orders
  2. Exit checker   — monitors SL/TP every EXIT_INTERVAL seconds

Also starts the FastAPI dashboard on DASHBOARD_PORT.

Usage:
  python bot.py              # paper trading (default)
  PAPER_TRADING=false python bot.py  # live trading (requires API keys)
"""

import asyncio
import signal as os_signal
import time
import pandas as pd
import uvicorn

import config
from exchange.paper   import PaperExchange
from exchange.binance import BinanceExchange
from alpha.engine     import AlphaEngine
from execution        import PortfolioRisk, OrderRouter
from dashboard.app    import app as dashboard_app, attach_bot
from utils.logger     import get_logger

logger = get_logger("nexus.bot")

LOOP_INTERVAL = 30   # seconds between strategy cycles
EXIT_INTERVAL = 10   # seconds between SL/TP checks


class NexusBot:

    def __init__(self):
        if config.PAPER_TRADING:
            self._exchange = PaperExchange(initial_capital=config.INITIAL_CAPITAL)
        else:
            self._exchange = BinanceExchange()

        self._alpha   = AlphaEngine()
        self._risk    = PortfolioRisk(initial_capital=config.INITIAL_CAPITAL)
        self._router  = OrderRouter(self._exchange, self._risk)
        self._running = False
        self._cycle   = 0

    async def run(self):
        logger.info("=" * 60)
        logger.info("  NEXUS TRADING SYSTEM — STARTING")
        logger.info(f"  Symbols  : {config.SYMBOLS}")
        logger.info(f"  Mode     : {'PAPER' if config.PAPER_TRADING else '*** LIVE ***'}")
        logger.info(f"  Capital  : ${config.INITIAL_CAPITAL:,.0f}")
        logger.info(f"  Dashboard: http://localhost:{config.DASHBOARD_PORT}")
        logger.info("=" * 60)

        self._running = True

        # Start WebSocket price stream for live mode
        if not config.PAPER_TRADING:
            await self._exchange.start_price_stream(config.SYMBOLS)

        await asyncio.gather(
            self._strategy_loop(),
            self._exit_checker_loop(),
        )

    async def _strategy_loop(self):
        while self._running:
            self._cycle += 1
            t0 = time.time()
            logger.info(f"── Cycle {self._cycle} ──────────────────────────────────")

            # Update funding rates
            for symbol in config.SYMBOLS:
                try:
                    rate = await self._exchange.get_funding_rate(symbol)
                    self._alpha.update_funding(rate)
                except Exception:
                    pass

            tasks = [self._process_symbol(sym) for sym in config.SYMBOLS]
            await asyncio.gather(*tasks, return_exceptions=True)

            summary = self._risk.summary()
            portfolio_val = await self._exchange.portfolio_value()
            logger.info(
                f"Portfolio=${portfolio_val:.2f} | "
                f"PnL={summary['total_pnl']:+.4f} | "
                f"Trades={summary['trade_count']} | "
                f"WinRate={summary['win_rate']:.1%} | "
                f"Drawdown={summary['drawdown_pct']:.1f}%"
            )

            elapsed = time.time() - t0
            await asyncio.sleep(max(0, LOOP_INTERVAL - elapsed))

    async def _process_symbol(self, symbol: str):
        try:
            candles = await self._exchange.get_candles(
                symbol, config.CANDLE_INTERVAL, config.CANDLE_LIMIT
            )
            ob = await self._exchange.get_orderbook(symbol, depth=20)

            df = pd.DataFrame(candles)
            for col in ("close", "open", "high", "low", "volume"):
                df[col] = df[col].astype(float)

            signal = self._alpha.evaluate(symbol, df, ob)
            await self._router.process(signal)

        except Exception as e:
            logger.error(f"[{symbol}] process_symbol error: {e}", exc_info=True)

    async def _exit_checker_loop(self):
        while self._running:
            try:
                await self._router.check_exits(config.SYMBOLS)
            except Exception as e:
                logger.error(f"Exit checker error: {e}")
            await asyncio.sleep(EXIT_INTERVAL)

    def stop(self):
        logger.info("Shutting down Nexus...")
        self._running = False


async def main():
    bot = NexusBot()
    attach_bot(bot)

    # Graceful shutdown
    loop = asyncio.get_event_loop()
    for sig in (os_signal.SIGINT, os_signal.SIGTERM):
        loop.add_signal_handler(sig, bot.stop)

    # Run dashboard + bot concurrently
    server_config = uvicorn.Config(
        dashboard_app,
        host="0.0.0.0",
        port=config.DASHBOARD_PORT,
        log_level="warning",
    )
    server = uvicorn.Server(server_config)

    try:
        await asyncio.gather(
            bot.run(),
            server.serve(),
        )
    finally:
        await bot._exchange.close()
        summary = bot._risk.summary()
        logger.info("=" * 60)
        logger.info("  NEXUS SHUTDOWN SUMMARY")
        logger.info(f"  Capital   : ${summary['capital']:,.2f}")
        logger.info(f"  Total PnL : {summary['total_pnl']:+.4f} USDT")
        logger.info(f"  Trades    : {summary['trade_count']}")
        logger.info(f"  Win Rate  : {summary['win_rate']:.1%}")
        logger.info(f"  Max DD    : {summary['drawdown_pct']:.1f}%")
        logger.info("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())
