"""
Data Access Objects (DAOs) for database operations.

Provides async CRUD operations for all database tables.
"""

import json
from datetime import datetime, timedelta
from typing import Optional, List, Dict, Any
import aiosqlite
from .connection import get_db_connection
from .models import (
    Stock, StockCreate, WatchlistItem, WatchlistItemCreate,
    AnalysisResult, PerformanceCheck, PerformanceStats,
    NewsItem, StockDataCache, Settings
)
import logging

logger = logging.getLogger(__name__)


# ===========================================
# Stock DAO
# ===========================================

class StockDAO:
    """Data Access Object for stocks table."""

    @staticmethod
    async def create(stock: StockCreate) -> Stock:
        """Create a new stock record."""
        async with get_db_connection() as db:
            await db.execute(
                """
                INSERT INTO stocks (ticker, company_name, sector, industry)
                VALUES (?, ?, ?, ?)
                ON CONFLICT(ticker) DO UPDATE SET
                    company_name = COALESCE(excluded.company_name, stocks.company_name),
                    sector = COALESCE(excluded.sector, stocks.sector),
                    industry = COALESCE(excluded.industry, stocks.industry),
                    last_updated = CURRENT_TIMESTAMP
                """,
                (stock.ticker, stock.company_name, stock.sector, stock.industry)
            )
            await db.commit()
            return await StockDAO.get_by_ticker(stock.ticker)

    @staticmethod
    async def get_by_ticker(ticker: str) -> Optional[Stock]:
        """Get a stock by ticker symbol."""
        async with get_db_connection() as db:
            async with db.execute(
                "SELECT * FROM stocks WHERE ticker = ?",
                (ticker.upper(),)
            ) as cursor:
                row = await cursor.fetchone()
                if row:
                    return Stock(
                        ticker=row['ticker'],
                        company_name=row['company_name'],
                        sector=row['sector'],
                        industry=row['industry'],
                        created_at=row['created_at'],
                        last_updated=row['last_updated']
                    )
                return None

    @staticmethod
    async def get_all() -> List[Stock]:
        """Get all stocks."""
        async with get_db_connection() as db:
            async with db.execute("SELECT * FROM stocks ORDER BY ticker") as cursor:
                rows = await cursor.fetchall()
                return [
                    Stock(
                        ticker=row['ticker'],
                        company_name=row['company_name'],
                        sector=row['sector'],
                        industry=row['industry'],
                        created_at=row['created_at'],
                        last_updated=row['last_updated']
                    )
                    for row in rows
                ]

    @staticmethod
    async def update(ticker: str, **kwargs) -> Optional[Stock]:
        """Update a stock record."""
        async with get_db_connection() as db:
            set_clauses = []
            values = []
            for key, value in kwargs.items():
                if key in ('company_name', 'sector', 'industry'):
                    set_clauses.append(f"{key} = ?")
                    values.append(value)

            if set_clauses:
                set_clauses.append("last_updated = CURRENT_TIMESTAMP")
                values.append(ticker.upper())
                await db.execute(
                    f"UPDATE stocks SET {', '.join(set_clauses)} WHERE ticker = ?",
                    values
                )
                await db.commit()

            return await StockDAO.get_by_ticker(ticker)

    @staticmethod
    async def delete(ticker: str) -> bool:
        """Delete a stock record."""
        async with get_db_connection() as db:
            result = await db.execute(
                "DELETE FROM stocks WHERE ticker = ?",
                (ticker.upper(),)
            )
            await db.commit()
            return result.rowcount > 0


# ===========================================
# Watchlist DAO
# ===========================================

class WatchlistDAO:
    """Data Access Object for watchlist table."""

    @staticmethod
    async def add(item: WatchlistItemCreate) -> WatchlistItem:
        """Add a stock to the watchlist."""
        ticker = item.ticker.upper()

        async with get_db_connection() as db:
            # Ensure stock exists
            existing = await StockDAO.get_by_ticker(ticker)
            if not existing:
                await StockDAO.create(StockCreate(ticker=ticker))

            # Add to watchlist
            await db.execute(
                "INSERT INTO watchlist (ticker) VALUES (?)",
                (ticker,)
            )
            await db.commit()

            # Get the created item
            return await WatchlistDAO.get_by_ticker(ticker)

    @staticmethod
    async def get_by_ticker(ticker: str) -> Optional[WatchlistItem]:
        """Get a watchlist item by ticker."""
        async with get_db_connection() as db:
            async with db.execute(
                """
                SELECT w.id, w.ticker, w.added_at, s.company_name
                FROM watchlist w
                LEFT JOIN stocks s ON w.ticker = s.ticker
                WHERE w.ticker = ?
                """,
                (ticker.upper(),)
            ) as cursor:
                row = await cursor.fetchone()
                if row:
                    # Get latest analysis for this ticker
                    async with db.execute(
                        """
                        SELECT recommendation, analysis_date
                        FROM analyses
                        WHERE ticker = ?
                        ORDER BY analysis_date DESC
                        LIMIT 1
                        """,
                        (ticker.upper(),)
                    ) as analysis_cursor:
                        analysis_row = await analysis_cursor.fetchone()

                    return WatchlistItem(
                        id=row['id'],
                        ticker=row['ticker'],
                        added_at=row['added_at'],
                        company_name=row['company_name'],
                        last_recommendation=analysis_row['recommendation'] if analysis_row else None,
                        last_analyzed=analysis_row['analysis_date'] if analysis_row else None
                    )
                return None

    @staticmethod
    async def get_all() -> List[WatchlistItem]:
        """Get all watchlist items with enriched data."""
        async with get_db_connection() as db:
            async with db.execute(
                """
                SELECT w.id, w.ticker, w.added_at, s.company_name
                FROM watchlist w
                LEFT JOIN stocks s ON w.ticker = s.ticker
                ORDER BY w.added_at DESC
                """
            ) as cursor:
                rows = await cursor.fetchall()

            items = []
            for row in rows:
                # Get latest analysis
                async with db.execute(
                    """
                    SELECT recommendation, analysis_date
                    FROM analyses
                    WHERE ticker = ?
                    ORDER BY analysis_date DESC
                    LIMIT 1
                    """,
                    (row['ticker'],)
                ) as analysis_cursor:
                    analysis_row = await analysis_cursor.fetchone()

                items.append(WatchlistItem(
                    id=row['id'],
                    ticker=row['ticker'],
                    added_at=row['added_at'],
                    company_name=row['company_name'],
                    last_recommendation=analysis_row['recommendation'] if analysis_row else None,
                    last_analyzed=analysis_row['analysis_date'] if analysis_row else None
                ))

            return items

    @staticmethod
    async def remove(ticker: str) -> bool:
        """Remove a stock from the watchlist."""
        async with get_db_connection() as db:
            result = await db.execute(
                "DELETE FROM watchlist WHERE ticker = ?",
                (ticker.upper(),)
            )
            await db.commit()
            return result.rowcount > 0

    @staticmethod
    async def exists(ticker: str) -> bool:
        """Check if a ticker is in the watchlist."""
        async with get_db_connection() as db:
            async with db.execute(
                "SELECT 1 FROM watchlist WHERE ticker = ?",
                (ticker.upper(),)
            ) as cursor:
                return await cursor.fetchone() is not None

    @staticmethod
    async def count() -> int:
        """Count items in watchlist."""
        async with get_db_connection() as db:
            async with db.execute("SELECT COUNT(*) FROM watchlist") as cursor:
                row = await cursor.fetchone()
                return row[0] if row else 0


# ===========================================
# Analysis DAO
# ===========================================

class AnalysisDAO:
    """Data Access Object for analyses table."""

    @staticmethod
    async def create(
        ticker: str,
        recommendation: str,
        confidence_level: str,
        analysis_style: str,
        reasoning: Optional[str] = None,
        price_at_analysis: Optional[float] = None,
        report_path: Optional[str] = None,
        llm_provider: Optional[str] = None,
        llm_model: Optional[str] = None
    ) -> AnalysisResult:
        """Create a new analysis record."""
        async with get_db_connection() as db:
            cursor = await db.execute(
                """
                INSERT INTO analyses (
                    ticker, recommendation, confidence_level, analysis_style,
                    reasoning, price_at_analysis, report_path, llm_provider, llm_model
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    ticker.upper(), recommendation, confidence_level, analysis_style,
                    reasoning, price_at_analysis, report_path, llm_provider, llm_model
                )
            )
            await db.commit()
            analysis_id = cursor.lastrowid

            return await AnalysisDAO.get_by_id(analysis_id)

    @staticmethod
    async def get_by_id(analysis_id: int) -> Optional[AnalysisResult]:
        """Get an analysis by ID."""
        async with get_db_connection() as db:
            async with db.execute(
                "SELECT * FROM analyses WHERE id = ?",
                (analysis_id,)
            ) as cursor:
                row = await cursor.fetchone()
                if row:
                    return AnalysisResult(
                        id=row['id'],
                        ticker=row['ticker'],
                        analysis_date=row['analysis_date'],
                        recommendation=row['recommendation'],
                        confidence_level=row['confidence_level'],
                        reasoning=row['reasoning'],
                        analysis_style=row['analysis_style'],
                        price_at_analysis=row['price_at_analysis'],
                        report_path=row['report_path'],
                        llm_provider=row['llm_provider'],
                        llm_model=row['llm_model']
                    )
                return None

    @staticmethod
    async def get_latest_by_ticker(ticker: str) -> Optional[AnalysisResult]:
        """Get the most recent analysis for a ticker."""
        async with get_db_connection() as db:
            async with db.execute(
                """
                SELECT * FROM analyses
                WHERE ticker = ?
                ORDER BY analysis_date DESC
                LIMIT 1
                """,
                (ticker.upper(),)
            ) as cursor:
                row = await cursor.fetchone()
                if row:
                    return AnalysisResult(
                        id=row['id'],
                        ticker=row['ticker'],
                        analysis_date=row['analysis_date'],
                        recommendation=row['recommendation'],
                        confidence_level=row['confidence_level'],
                        reasoning=row['reasoning'],
                        analysis_style=row['analysis_style'],
                        price_at_analysis=row['price_at_analysis'],
                        report_path=row['report_path'],
                        llm_provider=row['llm_provider'],
                        llm_model=row['llm_model']
                    )
                return None

    @staticmethod
    async def get_history(
        ticker: Optional[str] = None,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        limit: int = 100,
        offset: int = 0
    ) -> List[AnalysisResult]:
        """Get analysis history with optional filters."""
        async with get_db_connection() as db:
            query = "SELECT * FROM analyses WHERE 1=1"
            params = []

            if ticker:
                query += " AND ticker = ?"
                params.append(ticker.upper())

            if start_date:
                query += " AND analysis_date >= ?"
                params.append(start_date.isoformat())

            if end_date:
                query += " AND analysis_date <= ?"
                params.append(end_date.isoformat())

            query += " ORDER BY analysis_date DESC LIMIT ? OFFSET ?"
            params.extend([limit, offset])

            async with db.execute(query, params) as cursor:
                rows = await cursor.fetchall()
                return [
                    AnalysisResult(
                        id=row['id'],
                        ticker=row['ticker'],
                        analysis_date=row['analysis_date'],
                        recommendation=row['recommendation'],
                        confidence_level=row['confidence_level'],
                        reasoning=row['reasoning'],
                        analysis_style=row['analysis_style'],
                        price_at_analysis=row['price_at_analysis'],
                        report_path=row['report_path'],
                        llm_provider=row['llm_provider'],
                        llm_model=row['llm_model']
                    )
                    for row in rows
                ]

    @staticmethod
    async def count(
        ticker: Optional[str] = None,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None
    ) -> int:
        """Count analyses with optional filters."""
        async with get_db_connection() as db:
            query = "SELECT COUNT(*) FROM analyses WHERE 1=1"
            params = []

            if ticker:
                query += " AND ticker = ?"
                params.append(ticker.upper())

            if start_date:
                query += " AND analysis_date >= ?"
                params.append(start_date.isoformat())

            if end_date:
                query += " AND analysis_date <= ?"
                params.append(end_date.isoformat())

            async with db.execute(query, params) as cursor:
                row = await cursor.fetchone()
                return row[0] if row else 0

    @staticmethod
    async def get_unique_tickers() -> List[str]:
        """Get list of unique tickers with analyses."""
        async with get_db_connection() as db:
            async with db.execute(
                "SELECT DISTINCT ticker FROM analyses ORDER BY ticker"
            ) as cursor:
                rows = await cursor.fetchall()
                return [row['ticker'] for row in rows]


# ===========================================
# Performance DAO
# ===========================================

class PerformanceDAO:
    """Data Access Object for performance_checks table."""

    @staticmethod
    async def create(
        analysis_id: int,
        price_at_check: float,
        gain_loss_percent: Optional[float] = None,
        time_period_days: Optional[int] = None,
        outcome: Optional[str] = None
    ) -> PerformanceCheck:
        """Create a new performance check."""
        async with get_db_connection() as db:
            cursor = await db.execute(
                """
                INSERT INTO performance_checks (
                    analysis_id, price_at_check, gain_loss_percent,
                    time_period_days, outcome
                ) VALUES (?, ?, ?, ?, ?)
                """,
                (analysis_id, price_at_check, gain_loss_percent, time_period_days, outcome)
            )
            await db.commit()
            check_id = cursor.lastrowid

            return await PerformanceDAO.get_by_id(check_id)

    @staticmethod
    async def get_by_id(check_id: int) -> Optional[PerformanceCheck]:
        """Get a performance check by ID."""
        async with get_db_connection() as db:
            async with db.execute(
                "SELECT * FROM performance_checks WHERE id = ?",
                (check_id,)
            ) as cursor:
                row = await cursor.fetchone()
                if row:
                    return PerformanceCheck(
                        id=row['id'],
                        analysis_id=row['analysis_id'],
                        check_date=row['check_date'],
                        price_at_check=row['price_at_check'],
                        gain_loss_percent=row['gain_loss_percent'],
                        time_period_days=row['time_period_days'],
                        outcome=row['outcome']
                    )
                return None

    @staticmethod
    async def get_by_analysis(analysis_id: int) -> List[PerformanceCheck]:
        """Get all performance checks for an analysis."""
        async with get_db_connection() as db:
            async with db.execute(
                """
                SELECT * FROM performance_checks
                WHERE analysis_id = ?
                ORDER BY check_date DESC
                """,
                (analysis_id,)
            ) as cursor:
                rows = await cursor.fetchall()
                return [
                    PerformanceCheck(
                        id=row['id'],
                        analysis_id=row['analysis_id'],
                        check_date=row['check_date'],
                        price_at_check=row['price_at_check'],
                        gain_loss_percent=row['gain_loss_percent'],
                        time_period_days=row['time_period_days'],
                        outcome=row['outcome']
                    )
                    for row in rows
                ]

    @staticmethod
    async def get_stats() -> PerformanceStats:
        """Get aggregated performance statistics."""
        async with get_db_connection() as db:
            # Total analyses
            async with db.execute("SELECT COUNT(*) FROM analyses") as cursor:
                row = await cursor.fetchone()
                total_analyses = row[0] if row else 0

            # Performance check stats
            async with db.execute(
                """
                SELECT
                    COUNT(*) as total,
                    SUM(CASE WHEN outcome = 'Profitable' THEN 1 ELSE 0 END) as profitable,
                    SUM(CASE WHEN outcome = 'Loss' THEN 1 ELSE 0 END) as loss,
                    SUM(CASE WHEN outcome = 'Neutral' THEN 1 ELSE 0 END) as neutral,
                    AVG(gain_loss_percent) as avg_gain_loss
                FROM performance_checks
                """
            ) as cursor:
                row = await cursor.fetchone()

            if row:
                total_checked = row['total'] or 0
                profitable = row['profitable'] or 0
                loss = row['loss'] or 0
                neutral = row['neutral'] or 0
                avg_gain_loss = row['avg_gain_loss'] or 0.0

                success_rate = (profitable / total_checked * 100) if total_checked > 0 else 0.0
            else:
                total_checked = 0
                profitable = 0
                loss = 0
                neutral = 0
                avg_gain_loss = 0.0
                success_rate = 0.0

            return PerformanceStats(
                total_analyses=total_analyses,
                total_checked=total_checked,
                profitable_count=profitable,
                loss_count=loss,
                neutral_count=neutral,
                success_rate=success_rate,
                average_gain_loss=avg_gain_loss
            )


# ===========================================
# Stock Data Cache DAO
# ===========================================

class StockDataDAO:
    """Data Access Object for stock_data cache table."""

    @staticmethod
    async def save(
        ticker: str,
        data_type: str,
        data: Dict[str, Any],
        cache_hours: int = 1
    ) -> None:
        """Save stock data to cache."""
        expires_at = datetime.now() + timedelta(hours=cache_hours)
        data_json = json.dumps(data)

        async with get_db_connection() as db:
            # Delete existing cache for this ticker/type
            await db.execute(
                "DELETE FROM stock_data WHERE ticker = ? AND data_type = ?",
                (ticker.upper(), data_type)
            )

            # Insert new cache
            await db.execute(
                """
                INSERT INTO stock_data (ticker, data_type, data_json, cache_expires_at)
                VALUES (?, ?, ?, ?)
                """,
                (ticker.upper(), data_type, data_json, expires_at.isoformat())
            )
            await db.commit()

    @staticmethod
    async def get(ticker: str, data_type: str) -> Optional[Dict[str, Any]]:
        """Get cached stock data if not expired."""
        async with get_db_connection() as db:
            async with db.execute(
                """
                SELECT data_json, cache_expires_at FROM stock_data
                WHERE ticker = ? AND data_type = ?
                """,
                (ticker.upper(), data_type)
            ) as cursor:
                row = await cursor.fetchone()

                if row:
                    expires_at = datetime.fromisoformat(row['cache_expires_at'])
                    if expires_at > datetime.now():
                        return json.loads(row['data_json'])

                return None

    @staticmethod
    async def is_cached(ticker: str, data_type: str) -> bool:
        """Check if valid cache exists."""
        data = await StockDataDAO.get(ticker, data_type)
        return data is not None

    @staticmethod
    async def invalidate(ticker: str, data_type: Optional[str] = None) -> None:
        """Invalidate cache for a ticker."""
        async with get_db_connection() as db:
            if data_type:
                await db.execute(
                    "DELETE FROM stock_data WHERE ticker = ? AND data_type = ?",
                    (ticker.upper(), data_type)
                )
            else:
                await db.execute(
                    "DELETE FROM stock_data WHERE ticker = ?",
                    (ticker.upper(),)
                )
            await db.commit()

    @staticmethod
    async def cleanup_expired() -> int:
        """Remove expired cache entries."""
        async with get_db_connection() as db:
            result = await db.execute(
                "DELETE FROM stock_data WHERE cache_expires_at < ?",
                (datetime.now().isoformat(),)
            )
            await db.commit()
            return result.rowcount


# ===========================================
# News DAO
# ===========================================

class NewsDAO:
    """Data Access Object for news_data table."""

    @staticmethod
    async def save_many(ticker: str, news_items: List[Dict[str, Any]]) -> None:
        """Save multiple news items."""
        async with get_db_connection() as db:
            for item in news_items:
                await db.execute(
                    """
                    INSERT INTO news_data (ticker, headline, source, published_date, sentiment, url)
                    VALUES (?, ?, ?, ?, ?, ?)
                    """,
                    (
                        ticker.upper(),
                        item.get('headline'),
                        item.get('source'),
                        item.get('published_date'),
                        item.get('sentiment'),
                        item.get('url')
                    )
                )
            await db.commit()

    @staticmethod
    async def get_by_ticker(ticker: str, limit: int = 10) -> List[NewsItem]:
        """Get news items for a ticker."""
        async with get_db_connection() as db:
            async with db.execute(
                """
                SELECT * FROM news_data
                WHERE ticker = ?
                ORDER BY fetched_at DESC
                LIMIT ?
                """,
                (ticker.upper(), limit)
            ) as cursor:
                rows = await cursor.fetchall()
                return [
                    NewsItem(
                        id=row['id'],
                        ticker=row['ticker'],
                        headline=row['headline'],
                        source=row['source'],
                        published_date=row['published_date'],
                        sentiment=row['sentiment'],
                        url=row['url'],
                        fetched_at=row['fetched_at']
                    )
                    for row in rows
                ]


# ===========================================
# Settings DAO
# ===========================================

class SettingsDAO:
    """Data Access Object for settings table."""

    @staticmethod
    async def get(key: str, default: Optional[str] = None) -> Optional[str]:
        """Get a setting value."""
        async with get_db_connection() as db:
            async with db.execute(
                "SELECT value FROM settings WHERE key = ?",
                (key,)
            ) as cursor:
                row = await cursor.fetchone()
                return row['value'] if row else default

    @staticmethod
    async def set(key: str, value: str) -> None:
        """Set a setting value."""
        async with get_db_connection() as db:
            await db.execute(
                """
                INSERT INTO settings (key, value)
                VALUES (?, ?)
                ON CONFLICT(key) DO UPDATE SET
                    value = excluded.value,
                    updated_at = CURRENT_TIMESTAMP
                """,
                (key, value)
            )
            await db.commit()

    @staticmethod
    async def get_all() -> Dict[str, str]:
        """Get all settings."""
        async with get_db_connection() as db:
            async with db.execute("SELECT key, value FROM settings") as cursor:
                rows = await cursor.fetchall()
                return {row['key']: row['value'] for row in rows}

    @staticmethod
    async def delete(key: str) -> None:
        """Delete a setting."""
        async with get_db_connection() as db:
            await db.execute("DELETE FROM settings WHERE key = ?", (key,))
            await db.commit()
