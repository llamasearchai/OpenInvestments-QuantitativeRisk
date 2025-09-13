"""
Real-time market data integration and simulation.

Provides interfaces for:
- Live market data feeds (simulated)
- Historical data retrieval
- Data quality validation
- Price simulation engines
- Market microstructure analysis
"""

import asyncio
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Union, Callable, Any
from dataclasses import dataclass
from abc import ABC, abstractmethod
import random
import time

from ..core.logging import get_logger
from ..core.config import config

logger = get_logger(__name__)


@dataclass
class MarketData:
    """Market data container."""
    symbol: str
    timestamp: datetime
    price: float
    volume: Optional[int] = None
    bid: Optional[float] = None
    ask: Optional[float] = None
    high: Optional[float] = None
    low: Optional[float] = None
    open_price: Optional[float] = None
    close_price: Optional[float] = None


@dataclass
class OrderBook:
    """Order book representation."""
    symbol: str
    timestamp: datetime
    bids: List[tuple[float, int]]  # (price, quantity) pairs
    asks: List[tuple[float, int]]  # (price, quantity) pairs
    last_update: datetime


class MarketDataFeed(ABC):
    """Abstract base class for market data feeds."""

    @abstractmethod
    async def connect(self) -> bool:
        """Connect to the data feed."""
        pass

    @abstractmethod
    async def disconnect(self) -> None:
        """Disconnect from the data feed."""
        pass

    @abstractmethod
    async def subscribe(self, symbols: List[str]) -> None:
        """Subscribe to market data for given symbols."""
        pass

    @abstractmethod
    async def unsubscribe(self, symbols: List[str]) -> None:
        """Unsubscribe from market data for given symbols."""
        pass

    @abstractmethod
    async def get_current_price(self, symbol: str) -> Optional[MarketData]:
        """Get current price for a symbol."""
        pass

    @abstractmethod
    async def get_historical_data(
        self,
        symbol: str,
        start_date: datetime,
        end_date: datetime,
        interval: str = "1D"
    ) -> pd.DataFrame:
        """Get historical data for a symbol."""
        pass


class SimulatedMarketDataFeed(MarketDataFeed):
    """
    Simulated market data feed for testing and development.

    Generates realistic price movements using geometric Brownian motion
    and adds market microstructure effects.
    """

    def __init__(
        self,
        base_prices: Dict[str, float] = None,
        volatility: Dict[str, float] = None,
        correlation_matrix: Optional[np.ndarray] = None
    ):
        """
        Initialize simulated market data feed.

        Args:
            base_prices: Dictionary of symbol -> base price
            volatility: Dictionary of symbol -> annual volatility
            correlation_matrix: Correlation matrix for multi-asset simulation
        """
        self.base_prices = base_prices or {
            "AAPL": 150.0,
            "MSFT": 300.0,
            "GOOGL": 2500.0,
            "AMZN": 3200.0,
            "TSLA": 800.0,
            "SPY": 450.0,
            "QQQ": 380.0,
            "BTC": 50000.0
        }

        self.volatility = volatility or {symbol: 0.25 for symbol in self.base_prices.keys()}
        self.correlation_matrix = correlation_matrix

        # Current state
        self.connected = False
        self.subscribed_symbols = set()
        self.current_prices = {symbol: price for symbol, price in self.base_prices.items()}
        self.price_history = {symbol: [] for symbol in self.base_prices.keys()}

        # Simulation parameters
        self.drift = 0.08  # Annual drift
        self.time_step = 1.0 / 252.0  # Daily time steps
        self.last_update = datetime.now()

        # Market microstructure parameters
        self.spread_factor = 0.001  # Bid-ask spread as % of price
        self.volume_base = 1000000  # Base daily volume
        self.volume_volatility = 0.3  # Volume volatility

        self.logger = logger

    async def connect(self) -> bool:
        """Connect to simulated data feed."""
        self.logger.info("Connecting to simulated market data feed")
        await asyncio.sleep(0.1)  # Simulate connection time
        self.connected = True
        self.logger.info("Connected to simulated market data feed")
        return True

    async def disconnect(self) -> None:
        """Disconnect from simulated data feed."""
        self.logger.info("Disconnecting from simulated market data feed")
        self.connected = False
        self.logger.info("Disconnected from simulated market data feed")

    async def subscribe(self, symbols: List[str]) -> None:
        """Subscribe to market data for given symbols."""
        if not self.connected:
            raise ConnectionError("Not connected to market data feed")

        for symbol in symbols:
            if symbol not in self.base_prices:
                # Add new symbol with random parameters
                self.base_prices[symbol] = random.uniform(50, 500)
                self.volatility[symbol] = random.uniform(0.15, 0.4)
                self.current_prices[symbol] = self.base_prices[symbol]
                self.price_history[symbol] = []

        self.subscribed_symbols.update(symbols)
        self.logger.info(f"Subscribed to symbols: {symbols}")

    async def unsubscribe(self, symbols: List[str]) -> None:
        """Unsubscribe from market data for given symbols."""
        self.subscribed_symbols -= set(symbols)
        self.logger.info(f"Unsubscribed from symbols: {symbols}")

    async def get_current_price(self, symbol: str) -> Optional[MarketData]:
        """Get current price for a symbol."""
        if not self.connected or symbol not in self.current_prices:
            return None

        # Generate new price if enough time has passed
        now = datetime.now()
        if (now - self.last_update).seconds > 1:  # Update every second
            await self._update_prices()
            self.last_update = now

        price = self.current_prices[symbol]
        spread = price * self.spread_factor

        return MarketData(
            symbol=symbol,
            timestamp=now,
            price=price,
            volume=int(self.volume_base * (1 + random.gauss(0, self.volume_volatility))),
            bid=price - spread/2,
            ask=price + spread/2,
            high=max(self.price_history[symbol][-20:]) if self.price_history[symbol] else price,
            low=min(self.price_history[symbol][-20:]) if self.price_history[symbol] else price,
            open_price=self.price_history[symbol][0] if self.price_history[symbol] else price,
            close_price=price
        )

    async def get_historical_data(
        self,
        symbol: str,
        start_date: datetime,
        end_date: datetime,
        interval: str = "1D"
    ) -> pd.DataFrame:
        """Get historical data for a symbol."""
        if symbol not in self.base_prices:
            raise ValueError(f"Symbol {symbol} not found")

        # Calculate number of periods
        if interval == "1D":
            periods = (end_date - start_date).days
        elif interval == "1H":
            periods = int((end_date - start_date).total_seconds() / 3600)
        elif interval == "1m":
            periods = int((end_date - start_date).total_seconds() / 60)
        else:
            periods = (end_date - start_date).days

        # Generate historical prices
        prices = []
        current_price = self.base_prices[symbol]

        for i in range(periods):
            # Generate price movement
            drift_term = self.drift * self.time_step
            diffusion_term = self.volatility[symbol] * np.sqrt(self.time_step) * random.gauss(0, 1)

            price_change = current_price * (drift_term + diffusion_term)
            current_price += price_change
            current_price = max(current_price, 0.01)  # Prevent negative prices

            # Generate OHLC data
            volatility = self.volatility[symbol] * current_price
            high = current_price + abs(random.gauss(0, volatility * 0.1))
            low = current_price - abs(random.gauss(0, volatility * 0.1))
            open_price = current_price + random.gauss(0, volatility * 0.05)
            close_price = current_price

            # Generate volume
            volume = int(self.volume_base * (1 + random.gauss(0, self.volume_volatility)))

            prices.append({
                'timestamp': start_date + timedelta(days=i) if interval == "1D"
                           else start_date + timedelta(hours=i) if interval == "1H"
                           else start_date + timedelta(minutes=i),
                'symbol': symbol,
                'open': open_price,
                'high': max(open_price, high, close_price),
                'low': min(open_price, low, close_price),
                'close': close_price,
                'volume': volume
            })

        df = pd.DataFrame(prices)
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df.set_index('timestamp', inplace=True)

        self.logger.info(f"Generated historical data for {symbol}: {len(df)} periods")
        return df

    async def _update_prices(self) -> None:
        """Update current prices using GBM simulation."""
        for symbol in self.subscribed_symbols:
            if symbol in self.current_prices:
                current_price = self.current_prices[symbol]

                # Generate price movement
                drift_term = self.drift * self.time_step
                diffusion_term = self.volatility[symbol] * np.sqrt(self.time_step) * random.gauss(0, 1)

                price_change = current_price * (drift_term + diffusion_term)
                new_price = current_price + price_change
                new_price = max(new_price, 0.01)  # Prevent negative prices

                # Store in history
                self.price_history[symbol].append(current_price)
                if len(self.price_history[symbol]) > 1000:  # Keep last 1000 prices
                    self.price_history[symbol].pop(0)

                self.current_prices[symbol] = new_price

    async def get_order_book(self, symbol: str, depth: int = 10) -> Optional[OrderBook]:
        """Get order book for a symbol."""
        if not self.connected or symbol not in self.current_prices:
            return None

        price = self.current_prices[symbol]
        spread = price * self.spread_factor

        # Generate realistic order book
        bids = []
        asks = []

        for i in range(depth):
            bid_price = price - spread/2 - (i * spread / depth * 0.1)
            ask_price = price + spread/2 + (i * spread / depth * 0.1)

            bid_qty = int(random.expovariate(1.0 / 1000) + 100)
            ask_qty = int(random.expovariate(1.0 / 1000) + 100)

            bids.append((max(bid_price, 0.01), bid_qty))
            asks.append((ask_price, ask_qty))

        return OrderBook(
            symbol=symbol,
            timestamp=datetime.now(),
            bids=bids,
            asks=asks,
            last_update=datetime.now()
        )


class MarketDataManager:
    """
    Centralized market data management system.

    Provides unified interface for multiple data feeds and caching.
    """

    def __init__(self):
        self.feeds = {}
        self.cache = {}
        self.cache_expiry = 300  # 5 minutes
        self.logger = logger

        # Initialize default simulated feed
        self.add_feed("simulated", SimulatedMarketDataFeed())

    def add_feed(self, name: str, feed: MarketDataFeed) -> None:
        """Add a market data feed."""
        self.feeds[name] = feed
        self.logger.info(f"Added market data feed: {name}")

    async def get_price(self, symbol: str, feed_name: str = "simulated") -> Optional[MarketData]:
        """Get current price for a symbol."""
        if feed_name not in self.feeds:
            raise ValueError(f"Feed {feed_name} not found")

        feed = self.feeds[feed_name]

        # Check cache first
        cache_key = f"{feed_name}_{symbol}_price"
        if cache_key in self.cache:
            cached_data, timestamp = self.cache[cache_key]
            if (datetime.now() - timestamp).seconds < self.cache_expiry:
                return cached_data

        # Get fresh data
        data = await feed.get_current_price(symbol)

        if data:
            self.cache[cache_key] = (data, datetime.now())

        return data

    async def get_historical_prices(
        self,
        symbols: List[str],
        start_date: datetime,
        end_date: datetime,
        interval: str = "1D",
        feed_name: str = "simulated"
    ) -> Dict[str, pd.DataFrame]:
        """Get historical prices for multiple symbols."""
        if feed_name not in self.feeds:
            raise ValueError(f"Feed {feed_name} not found")

        feed = self.feeds[feed_name]
        results = {}

        for symbol in symbols:
            # Check cache
            cache_key = f"{feed_name}_{symbol}_{start_date}_{end_date}_{interval}"
            if cache_key in self.cache:
                cached_data, timestamp = self.cache[cache_key]
                if (datetime.now() - timestamp).seconds < self.cache_expiry:
                    results[symbol] = cached_data
                    continue

            # Get fresh data
            data = await feed.get_historical_data(symbol, start_date, end_date, interval)
            results[symbol] = data

            # Cache result
            self.cache[cache_key] = (data, datetime.now())

        return results

    async def subscribe_symbols(self, symbols: List[str], feed_name: str = "simulated") -> None:
        """Subscribe to real-time data for symbols."""
        if feed_name not in self.feeds:
            raise ValueError(f"Feed {feed_name} not found")

        await self.feeds[feed_name].subscribe(symbols)

    def validate_data_quality(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Validate data quality and return quality metrics."""
        quality_report = {
            "total_rows": len(data),
            "missing_values": data.isnull().sum().to_dict(),
            "duplicate_rows": data.duplicated().sum(),
            "outliers": {},
            "data_types": data.dtypes.to_dict()
        }

        # Check for outliers in numeric columns
        numeric_cols = data.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            if col in data.columns:
                series = data[col].dropna()
                if len(series) > 0:
                    q1 = series.quantile(0.25)
                    q3 = series.quantile(0.75)
                    iqr = q3 - q1
                    outliers = ((series < (q1 - 1.5 * iqr)) | (series > (q3 + 1.5 * iqr))).sum()
                    quality_report["outliers"][col] = outliers

        return quality_report

    async def get_market_snapshot(self, symbols: List[str], feed_name: str = "simulated") -> Dict[str, MarketData]:
        """Get current market snapshot for multiple symbols."""
        snapshot = {}

        for symbol in symbols:
            data = await self.get_price(symbol, feed_name)
            if data:
                snapshot[symbol] = data

        self.logger.info(f"Retrieved market snapshot for {len(snapshot)} symbols")
        return snapshot


# Global market data manager instance
market_data_manager = MarketDataManager()


async def get_market_data_manager() -> MarketDataManager:
    """Get the global market data manager instance."""
    return market_data_manager
