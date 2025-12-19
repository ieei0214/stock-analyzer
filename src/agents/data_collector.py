"""
Data Collector Agent.

This agent is responsible for fetching comprehensive stock data from yfinance,
including price history, financial metrics, technical indicators, and news.
"""

import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, List
import logging
import json

logger = logging.getLogger(__name__)


class DataCollectorAgent:
    """
    Agent for collecting stock data from various sources.

    Fetches:
    - Price history (1 year)
    - Financial metrics (P/E, EPS, Revenue Growth, etc.)
    - Technical indicators (MA, RSI, MACD)
    - Analyst ratings and price targets
    - News headlines
    - Insider trading activity
    """

    def __init__(self, ticker: str):
        """Initialize the data collector for a specific ticker."""
        self.ticker = ticker.upper()
        self.stock = yf.Ticker(self.ticker)
        self._info = None
        self._history = None

    def _get_info(self) -> Dict[str, Any]:
        """Get stock info with caching."""
        if self._info is None:
            try:
                self._info = self.stock.info
            except Exception as e:
                logger.error(f"Failed to fetch info for {self.ticker}: {e}")
                self._info = {}
        return self._info

    def _get_history(self, period: str = "1y") -> pd.DataFrame:
        """Get price history with caching."""
        if self._history is None:
            try:
                self._history = self.stock.history(period=period)
            except Exception as e:
                logger.error(f"Failed to fetch history for {self.ticker}: {e}")
                self._history = pd.DataFrame()
        return self._history

    def validate_ticker(self) -> bool:
        """
        Validate that the ticker is a real stock.

        Returns True if valid, False otherwise.
        """
        try:
            info = self._get_info()
            # Check if we have meaningful data
            # yfinance returns empty or minimal data for invalid tickers
            if not info:
                return False

            # Check for key indicators of a real stock
            has_price = info.get('regularMarketPrice') or info.get('previousClose')
            has_name = info.get('shortName') or info.get('longName')

            return bool(has_price or has_name)
        except Exception as e:
            logger.error(f"Error validating ticker {self.ticker}: {e}")
            return False

    def get_company_info(self) -> Dict[str, Any]:
        """Get basic company information."""
        info = self._get_info()

        return {
            "ticker": self.ticker,
            "company_name": info.get("shortName") or info.get("longName", "Unknown"),
            "sector": info.get("sector", "Unknown"),
            "industry": info.get("industry", "Unknown"),
            "website": info.get("website"),
            "description": info.get("longBusinessSummary", ""),
            "country": info.get("country"),
            "employees": info.get("fullTimeEmployees"),
            "market_cap": info.get("marketCap"),
            "exchange": info.get("exchange"),
        }

    def get_current_price(self) -> Dict[str, Any]:
        """Get current price and basic trading info."""
        info = self._get_info()

        current_price = info.get("regularMarketPrice") or info.get("currentPrice") or info.get("previousClose")
        previous_close = info.get("regularMarketPreviousClose") or info.get("previousClose")

        change = None
        change_percent = None
        if current_price and previous_close:
            change = current_price - previous_close
            change_percent = (change / previous_close) * 100

        return {
            "current_price": current_price,
            "previous_close": previous_close,
            "change": change,
            "change_percent": change_percent,
            "open": info.get("regularMarketOpen") or info.get("open"),
            "high": info.get("regularMarketDayHigh") or info.get("dayHigh"),
            "low": info.get("regularMarketDayLow") or info.get("dayLow"),
            "volume": info.get("regularMarketVolume") or info.get("volume"),
            "avg_volume": info.get("averageVolume"),
            "fifty_two_week_high": info.get("fiftyTwoWeekHigh"),
            "fifty_two_week_low": info.get("fiftyTwoWeekLow"),
            "market_cap": info.get("marketCap"),
        }

    def get_price_history(self, period: str = "1y") -> Dict[str, Any]:
        """
        Get historical price data.

        Returns price history as lists for charting.
        """
        history = self._get_history(period)

        if history.empty:
            return {
                "dates": [],
                "open": [],
                "high": [],
                "low": [],
                "close": [],
                "volume": [],
            }

        # Convert to serializable format
        history_reset = history.reset_index()

        return {
            "dates": [d.strftime("%Y-%m-%d") for d in history_reset["Date"]],
            "open": history_reset["Open"].fillna(0).tolist(),
            "high": history_reset["High"].fillna(0).tolist(),
            "low": history_reset["Low"].fillna(0).tolist(),
            "close": history_reset["Close"].fillna(0).tolist(),
            "volume": history_reset["Volume"].fillna(0).astype(int).tolist(),
        }

    def get_financial_metrics(self) -> Dict[str, Any]:
        """Get key financial metrics for fundamental analysis."""
        info = self._get_info()

        return {
            # Valuation metrics
            "pe_ratio": info.get("trailingPE"),
            "forward_pe": info.get("forwardPE"),
            "peg_ratio": info.get("pegRatio"),
            "price_to_book": info.get("priceToBook"),
            "price_to_sales": info.get("priceToSalesTrailing12Months"),

            # Earnings metrics
            "eps": info.get("trailingEps"),
            "forward_eps": info.get("forwardEps"),
            "earnings_growth": info.get("earningsGrowth"),

            # Revenue metrics
            "revenue": info.get("totalRevenue"),
            "revenue_growth": info.get("revenueGrowth"),
            "revenue_per_share": info.get("revenuePerShare"),

            # Profitability metrics
            "gross_margin": info.get("grossMargins"),
            "operating_margin": info.get("operatingMargins"),
            "profit_margin": info.get("profitMargins"),
            "return_on_equity": info.get("returnOnEquity"),
            "return_on_assets": info.get("returnOnAssets"),

            # Debt metrics
            "debt_to_equity": info.get("debtToEquity"),
            "current_ratio": info.get("currentRatio"),
            "quick_ratio": info.get("quickRatio"),

            # Dividend metrics
            "dividend_yield": info.get("dividendYield"),
            "dividend_rate": info.get("dividendRate"),
            "payout_ratio": info.get("payoutRatio"),

            # Other
            "beta": info.get("beta"),
            "book_value": info.get("bookValue"),
            "enterprise_value": info.get("enterpriseValue"),
        }

    def calculate_technical_indicators(self) -> Dict[str, Any]:
        """Calculate technical indicators from price history."""
        history = self._get_history("1y")

        if history.empty or len(history) < 50:
            return {
                "ma_50": None,
                "ma_200": None,
                "rsi_14": None,
                "macd": None,
                "macd_signal": None,
                "macd_histogram": None,
                "bollinger_upper": None,
                "bollinger_lower": None,
                "ma_cross_signal": None,
            }

        close = history["Close"]

        # Moving averages
        ma_50 = close.rolling(window=50).mean()
        ma_200 = close.rolling(window=200).mean() if len(close) >= 200 else None

        # RSI (14-period)
        delta = close.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))

        # MACD (12, 26, 9)
        exp12 = close.ewm(span=12, adjust=False).mean()
        exp26 = close.ewm(span=26, adjust=False).mean()
        macd = exp12 - exp26
        macd_signal = macd.ewm(span=9, adjust=False).mean()
        macd_histogram = macd - macd_signal

        # Bollinger Bands (20-period, 2 std)
        bb_middle = close.rolling(window=20).mean()
        bb_std = close.rolling(window=20).std()
        bb_upper = bb_middle + (bb_std * 2)
        bb_lower = bb_middle - (bb_std * 2)

        # Get latest values
        latest_close = float(close.iloc[-1])
        latest_ma_50 = float(ma_50.iloc[-1]) if not pd.isna(ma_50.iloc[-1]) else None
        latest_ma_200 = float(ma_200.iloc[-1]) if ma_200 is not None and not pd.isna(ma_200.iloc[-1]) else None
        latest_rsi = float(rsi.iloc[-1]) if not pd.isna(rsi.iloc[-1]) else None
        latest_macd = float(macd.iloc[-1]) if not pd.isna(macd.iloc[-1]) else None
        latest_macd_signal = float(macd_signal.iloc[-1]) if not pd.isna(macd_signal.iloc[-1]) else None
        latest_macd_histogram = float(macd_histogram.iloc[-1]) if not pd.isna(macd_histogram.iloc[-1]) else None

        # Moving average cross signal
        ma_cross_signal = None
        if latest_ma_50 and latest_ma_200:
            if latest_ma_50 > latest_ma_200:
                ma_cross_signal = "Bullish (Golden Cross)"
            else:
                ma_cross_signal = "Bearish (Death Cross)"

        # RSI signal
        rsi_signal = None
        if latest_rsi:
            if latest_rsi > 70:
                rsi_signal = "Overbought"
            elif latest_rsi < 30:
                rsi_signal = "Oversold"
            else:
                rsi_signal = "Neutral"

        return {
            "current_price": latest_close,
            "ma_50": latest_ma_50,
            "ma_200": latest_ma_200,
            "ma_cross_signal": ma_cross_signal,
            "rsi_14": latest_rsi,
            "rsi_signal": rsi_signal,
            "macd": latest_macd,
            "macd_signal": latest_macd_signal,
            "macd_histogram": latest_macd_histogram,
            "bollinger_upper": float(bb_upper.iloc[-1]) if not pd.isna(bb_upper.iloc[-1]) else None,
            "bollinger_middle": float(bb_middle.iloc[-1]) if not pd.isna(bb_middle.iloc[-1]) else None,
            "bollinger_lower": float(bb_lower.iloc[-1]) if not pd.isna(bb_lower.iloc[-1]) else None,
            # Historical data for charts
            "ma_50_history": [float(x) if not pd.isna(x) else None for x in ma_50.tail(100).tolist()],
            "ma_200_history": [float(x) if not pd.isna(x) else None for x in (ma_200.tail(100).tolist() if ma_200 is not None else [])],
            "rsi_history": [float(x) if not pd.isna(x) else None for x in rsi.tail(100).tolist()],
            "macd_history": [float(x) if not pd.isna(x) else None for x in macd.tail(100).tolist()],
            "macd_signal_history": [float(x) if not pd.isna(x) else None for x in macd_signal.tail(100).tolist()],
        }

    def get_analyst_ratings(self) -> Dict[str, Any]:
        """Get analyst recommendations and price targets."""
        info = self._get_info()

        try:
            recommendations = self.stock.recommendations
            if recommendations is not None and not recommendations.empty:
                recent_recs = recommendations.tail(10)
                rec_list = []
                for idx, row in recent_recs.iterrows():
                    rec_list.append({
                        "date": idx.strftime("%Y-%m-%d") if hasattr(idx, 'strftime') else str(idx),
                        "firm": row.get("Firm", "Unknown"),
                        "to_grade": row.get("To Grade", row.get("toGrade", "")),
                        "from_grade": row.get("From Grade", row.get("fromGrade", "")),
                        "action": row.get("Action", row.get("action", "")),
                    })
            else:
                rec_list = []
        except Exception as e:
            logger.warning(f"Failed to fetch recommendations: {e}")
            rec_list = []

        return {
            "recommendation_key": info.get("recommendationKey"),
            "recommendation_mean": info.get("recommendationMean"),
            "number_of_analysts": info.get("numberOfAnalystOpinions"),
            "target_high": info.get("targetHighPrice"),
            "target_low": info.get("targetLowPrice"),
            "target_mean": info.get("targetMeanPrice"),
            "target_median": info.get("targetMedianPrice"),
            "recent_recommendations": rec_list,
        }

    def _analyze_headline_sentiment(self, headline: str) -> str:
        """
        Analyze sentiment of a news headline using keyword matching.
        Returns 'Positive', 'Negative', or 'Neutral'.
        """
        if not headline:
            return "Neutral"

        headline_lower = headline.lower()

        # Positive keywords (financial/stock context)
        positive_keywords = [
            'surge', 'soar', 'rally', 'gain', 'jump', 'rise', 'up', 'high', 'record',
            'beat', 'exceed', 'outperform', 'strong', 'growth', 'profit', 'bullish',
            'upgrade', 'buy', 'breakout', 'boost', 'win', 'success', 'best', 'top',
            'positive', 'optimistic', 'recommend', 'opportunity', 'momentum',
            'advance', 'improve', 'recover', 'rebound', 'milestone', 'innovation'
        ]

        # Negative keywords (financial/stock context)
        negative_keywords = [
            'fall', 'drop', 'decline', 'plunge', 'crash', 'tumble', 'slide', 'sink',
            'miss', 'loss', 'lose', 'weak', 'bearish', 'downgrade', 'sell', 'warning',
            'risk', 'concern', 'worry', 'fear', 'trouble', 'problem', 'crisis',
            'cut', 'slash', 'layoff', 'lawsuit', 'investigation', 'fraud', 'scandal',
            'worse', 'worst', 'struggle', 'fail', 'bankruptcy', 'debt', 'default'
        ]

        # Count matches
        positive_count = sum(1 for word in positive_keywords if word in headline_lower)
        negative_count = sum(1 for word in negative_keywords if word in headline_lower)

        # Determine sentiment
        if positive_count > negative_count:
            return "Positive"
        elif negative_count > positive_count:
            return "Negative"
        else:
            return "Neutral"

    def get_news(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Get recent news headlines for the stock."""
        try:
            news = self.stock.news
            if not news:
                return []

            news_list = []
            for item in news[:limit]:
                # Handle different yfinance versions
                if not isinstance(item, dict):
                    continue

                # Safely get content dict
                content = item.get("content") if isinstance(item.get("content"), dict) else {}

                # Try to get title from various possible locations
                title = item.get("title", "")
                if not title and content:
                    title = content.get("title", "")

                # Publisher
                publisher = item.get("publisher", "")
                if not publisher and content:
                    provider = content.get("provider")
                    if isinstance(provider, dict):
                        publisher = provider.get("displayName", "")

                # Link
                link = item.get("link", "") or item.get("url", "")
                if not link and content:
                    click_through = content.get("clickThroughUrl")
                    if isinstance(click_through, dict):
                        link = click_through.get("url", "")

                # Timestamp
                pub_time = item.get("providerPublishTime")
                if not pub_time and content:
                    pub_time = content.get("pubDate")

                published_at = None
                if pub_time:
                    try:
                        if isinstance(pub_time, (int, float)):
                            published_at = datetime.fromtimestamp(pub_time).isoformat()
                        elif isinstance(pub_time, str):
                            published_at = pub_time
                    except Exception:
                        pass

                # Thumbnail
                thumbnail = None
                thumb_data = item.get("thumbnail")
                if not thumb_data and content:
                    thumb_data = content.get("thumbnail")

                if isinstance(thumb_data, dict):
                    resolutions = thumb_data.get("resolutions", [])
                    if resolutions and len(resolutions) > 0:
                        thumbnail = resolutions[0].get("url")

                # Analyze sentiment based on headline keywords
                sentiment = self._analyze_headline_sentiment(title)

                news_list.append({
                    "title": title or "",
                    "publisher": publisher or "",
                    "link": link or "",
                    "published_at": published_at,
                    "type": item.get("type", ""),
                    "thumbnail": thumbnail,
                    "sentiment": sentiment,
                })

            return news_list
        except Exception as e:
            logger.warning(f"Failed to fetch news: {e}")
            return []

    def get_insider_trading(self) -> List[Dict[str, Any]]:
        """Get insider trading activity."""
        try:
            insider_transactions = self.stock.insider_transactions
            if insider_transactions is None or insider_transactions.empty:
                return []

            transactions = []
            for idx, row in insider_transactions.head(20).iterrows():
                # Handle shares - check for nan values
                shares_val = row.get("Shares", row.get("shares", 0))
                if pd.isna(shares_val):
                    shares = 0
                else:
                    try:
                        shares = int(shares_val)
                    except (ValueError, TypeError):
                        shares = 0

                # Handle value - check for nan values
                value_val = row.get("Value", row.get("value", 0))
                if pd.isna(value_val):
                    value = 0.0
                else:
                    try:
                        value = float(value_val)
                    except (ValueError, TypeError):
                        value = 0.0

                transactions.append({
                    "insider": str(row.get("Insider Trading", row.get("insider", "Unknown")) or "Unknown"),
                    "relation": str(row.get("Relationship", row.get("relationship", "")) or ""),
                    "date": str(row.get("Start Date", row.get("startDate", "")) or ""),
                    "transaction": str(row.get("Transaction", row.get("transaction", "")) or ""),
                    "shares": shares,
                    "value": value,
                })

            return transactions
        except Exception as e:
            logger.warning(f"Failed to fetch insider trading: {e}")
            return []

    def collect_all_data(self) -> Dict[str, Any]:
        """
        Collect all stock data in a single call.

        Returns a comprehensive data structure with all available information.
        """
        logger.info(f"Collecting all data for {self.ticker}")

        # Validate ticker first
        if not self.validate_ticker():
            raise ValueError(f"Invalid ticker symbol: {self.ticker}")

        return {
            "ticker": self.ticker,
            "collected_at": datetime.now().isoformat(),
            "company": self.get_company_info(),
            "price": self.get_current_price(),
            "history": self.get_price_history(),
            "financials": self.get_financial_metrics(),
            "technicals": self.calculate_technical_indicators(),
            "analyst": self.get_analyst_ratings(),
            "news": self.get_news(),
            "insider_trading": self.get_insider_trading(),
        }


async def collect_stock_data(ticker: str) -> Dict[str, Any]:
    """
    Async wrapper to collect stock data.

    This function can be used in async contexts while
    the underlying yfinance calls are synchronous.
    """
    collector = DataCollectorAgent(ticker)
    return collector.collect_all_data()


def validate_stock_ticker(ticker: str) -> bool:
    """Quick validation of a stock ticker."""
    collector = DataCollectorAgent(ticker)
    return collector.validate_ticker()
