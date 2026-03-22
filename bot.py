#!/usr/bin/env python3
"""
bot.py — Nexus Trading System entry point.

Async tasks:
  1. Strategy loop  — candles + OB → alpha engine → order router (every 30s)
  2. Exit checker   — SL/TP + order timeout (every 10s)
  3. Dashboard      — FastAPI on DASHBOARD_PORT

Improvements over v1:
  - Config validation on startup
  - Circuit breaker per symbol
  - Order timeout cancellation
  - Volatility-adjusted position sizing (df passed to router)
  - Async locks in risk manager
  - Per-strategy performance metrics exposed to dashboard
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
from utils.circuit_breaker import CircuitBreaker, CircuitBreakerOpen

logger = get_logger("nexus.bot")

LOOP_INTERVAL = 30
EXIT_INTERVAL = 10


class NexusBot:

    def __init__(self):
        config.validate()   # fail fast on bad config

        self._exchange = (
            PaperExchange(initial_capital=config.INITIAL_CAPITAL)
            if config.PAPER_TRADING
            else BinanceExchange()
        )
        self._alpha   = AlphaEngine()
        self._risk    = PortfolioRisk(initial_capital=config.INITIAL_CAPITAL)
        self._router  = OrderRouter(self._exchange, self._risk)
        self._running = False
        self._cycle   = 0

        # Per-symbol circuit breakers
        self._cb: dict[str, CircuitBreaker] = {
            sym: CircuitBreaker(sym, max_failures=config.CIRCUIT_BREAKER_ERRORS,
                                cooldown=config.CIRCUIT_BREAKER_COOLDOWN)
            for sym in config.SYMBOLS
        }

    async def run(self):
        logger.info("=" * 60)
        logger.info("  NEXUS TRADING SYSTEM — STARTING")
        logger.info(f"  Symbols  : {config.SYMBOLS}")
        logger.info(f"  Mode     : {'PAPER' if config.PAPER_TRADING else '*** LIVE ***'}")
        logger.info(f"  Capital  : ${config.INITIAL_CAPITAL:,.0f}")
        logger.info(f"  Dashboard: http://localhost:{config.DASHBOARD_PORT}")
        logger.info("=" * 60)

        self._running = True

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
                f"Drawdown={summary['drawdown_pct']:.1f}% | "
                f"Halted={summary['halted']}"
            )

            elapsed = time.time() - t0
            await asyncio.sleep(max(0, LOOP_INTERVAL - elapsed))

    async def _process_symbol(self, symbol: str):
        cb = self._cb[symbol]
        try:
            async with cb:
                candles = await self._exchange.get_candles(
                    symbol, config.CANDLE_INTERVAL, config.CANDLE_LIMIT
                )
                ob = await self._exchange.get_orderbook(symbol, depth=20)

                df = pd.DataFrame(candles)
                for col in ("close", "open", "high", "low", "volume"):
                    df[col] = df[col].astype(float)

                signal = self._alpha.evaluate(symbol, df, ob)
                await self._router.process(signal, df=df)

        except CircuitBreakerOpen as e:
            logger.warning(str(e))
        except Exception as e:
            logger.error(f"[{symbol}] process_symbol error: {e}", exc_info=True)

    async def _exit_checker_loop(self):
        while self._running:
            try:
                await self._router.check_exits(config.SYMBOLS)
                await self._router.cancel_timed_out_orders()
                if config.PAPER_TRADING:
                    await self._exchange.check_limit_fills()
            except Exception as e:
                logger.error(f"Exit checker error: {e}")
            await asyncio.sleep(EXIT_INTERVAL)

    def stop(self):
        logger.info("Shutting down Nexus...")
        self._running = False

    def circuit_breaker_status(self) -> list[dict]:
        return [cb.status() for cb in self._cb.values()]


async def main():
    bot = NexusBot()
    attach_bot(bot)

    loop = asyncio.get_event_loop()
    for sig in (os_signal.SIGINT, os_signal.SIGTERM):
        loop.add_signal_handler(sig, bot.stop)

    server_config = uvicorn.Config(
        dashboard_app,
        host="0.0.0.0",
        port=config.DASHBOARD_PORT,
        log_level="warning",
    )
    server = uvicorn.Server(server_config)

    try:
        await asyncio.gather(bot.run(), server.serve())
    finally:
        await bot._exchange.close()
        summary = bot._risk.summary()
        logger.info("=" * 60)
        logger.info("  NEXUS SHUTDOWN SUMMARY")
        logger.info(f"  Capital   : ${summary['capital']:,.2f}")
        logger.info(f"  Total PnL : {summary['total_pnl']:+.4f} USDT")
        logger.info(f"  Trades    : {summary['trade_count']}")
        logger.info(f"  Win Rate  : {summary['win_rate']:.1%}")
        logger.info(f"  Avg Hold  : {summary['avg_hold_min']:.1f} min")
        logger.info(f"  Max DD    : {summary['drawdown_pct']:.1f}%")
        logger.info("=" * 60)
        # Print per-strategy breakdown
        for strat, metrics in bot._risk.strategy_performance().items():
            logger.info(f"  [{strat}] {metrics}")


if __name__ == "__main__":
    asyncio.run(main())
