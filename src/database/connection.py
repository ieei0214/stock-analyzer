"""
Database connection and initialization module.

Provides SQLite connection management with async support using aiosqlite.
"""

import aiosqlite
import sqlite3
from pathlib import Path
from contextlib import asynccontextmanager
from typing import Optional
import logging

# Configure logging
logger = logging.getLogger(__name__)

# Database path
DATABASE_DIR = Path(__file__).parent.parent.parent / "data"
DATABASE_PATH = DATABASE_DIR / "stock_analyzer.db"


def get_sync_connection() -> sqlite3.Connection:
    """Get a synchronous database connection (for initialization)."""
    DATABASE_DIR.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(DATABASE_PATH))
    conn.row_factory = sqlite3.Row
    return conn


@asynccontextmanager
async def get_db_connection():
    """
    Async context manager for database connections.

    Usage:
        async with get_db_connection() as db:
            await db.execute("SELECT * FROM stocks")
    """
    DATABASE_DIR.mkdir(parents=True, exist_ok=True)
    db = await aiosqlite.connect(str(DATABASE_PATH))
    db.row_factory = aiosqlite.Row
    try:
        yield db
    finally:
        await db.close()


async def init_database():
    """Initialize the database with all required tables."""
    logger.info("Initializing database...")

    async with get_db_connection() as db:
        # Enable foreign keys
        await db.execute("PRAGMA foreign_keys = ON")

        # Create stocks table
        await db.execute("""
            CREATE TABLE IF NOT EXISTS stocks (
                ticker TEXT PRIMARY KEY,
                company_name TEXT,
                sector TEXT,
                industry TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)

        # Create watchlist table
        await db.execute("""
            CREATE TABLE IF NOT EXISTS watchlist (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                ticker TEXT NOT NULL,
                added_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (ticker) REFERENCES stocks(ticker) ON DELETE CASCADE,
                UNIQUE(ticker)
            )
        """)

        # Create stock_data table (for caching)
        await db.execute("""
            CREATE TABLE IF NOT EXISTS stock_data (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                ticker TEXT NOT NULL,
                data_type TEXT NOT NULL,
                data_json TEXT NOT NULL,
                fetched_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                cache_expires_at TIMESTAMP,
                FOREIGN KEY (ticker) REFERENCES stocks(ticker) ON DELETE CASCADE
            )
        """)

        # Create analyses table
        await db.execute("""
            CREATE TABLE IF NOT EXISTS analyses (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                ticker TEXT NOT NULL,
                analysis_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                recommendation TEXT NOT NULL CHECK (recommendation IN ('Buy', 'Sell', 'Hold')),
                confidence_level TEXT NOT NULL CHECK (confidence_level IN ('High', 'Medium', 'Low')),
                reasoning TEXT,
                analysis_style TEXT NOT NULL CHECK (analysis_style IN ('Conservative', 'Aggressive')),
                price_at_analysis REAL,
                report_path TEXT,
                llm_provider TEXT,
                llm_model TEXT,
                FOREIGN KEY (ticker) REFERENCES stocks(ticker) ON DELETE CASCADE
            )
        """)

        # Create performance_checks table
        await db.execute("""
            CREATE TABLE IF NOT EXISTS performance_checks (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                analysis_id INTEGER NOT NULL,
                check_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                price_at_check REAL NOT NULL,
                gain_loss_percent REAL,
                time_period_days INTEGER,
                outcome TEXT CHECK (outcome IN ('Profitable', 'Loss', 'Neutral')),
                FOREIGN KEY (analysis_id) REFERENCES analyses(id) ON DELETE CASCADE
            )
        """)

        # Create news_data table
        await db.execute("""
            CREATE TABLE IF NOT EXISTS news_data (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                ticker TEXT NOT NULL,
                headline TEXT NOT NULL,
                source TEXT,
                published_date TIMESTAMP,
                sentiment TEXT CHECK (sentiment IN ('Positive', 'Negative', 'Neutral')),
                url TEXT,
                fetched_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (ticker) REFERENCES stocks(ticker) ON DELETE CASCADE
            )
        """)

        # Create settings table
        await db.execute("""
            CREATE TABLE IF NOT EXISTS settings (
                key TEXT PRIMARY KEY,
                value TEXT NOT NULL,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)

        # Create indexes for better performance
        await db.execute("""
            CREATE INDEX IF NOT EXISTS idx_watchlist_ticker ON watchlist(ticker)
        """)

        await db.execute("""
            CREATE INDEX IF NOT EXISTS idx_stock_data_ticker_type ON stock_data(ticker, data_type)
        """)

        await db.execute("""
            CREATE INDEX IF NOT EXISTS idx_stock_data_expires ON stock_data(cache_expires_at)
        """)

        await db.execute("""
            CREATE INDEX IF NOT EXISTS idx_analyses_ticker ON analyses(ticker)
        """)

        await db.execute("""
            CREATE INDEX IF NOT EXISTS idx_analyses_date ON analyses(analysis_date)
        """)

        await db.execute("""
            CREATE INDEX IF NOT EXISTS idx_performance_analysis ON performance_checks(analysis_id)
        """)

        await db.execute("""
            CREATE INDEX IF NOT EXISTS idx_news_ticker ON news_data(ticker)
        """)

        await db.commit()

    logger.info("Database initialized successfully")


async def check_database_exists() -> bool:
    """Check if the database file exists."""
    return DATABASE_PATH.exists()


async def get_database_stats() -> dict:
    """Get statistics about the database."""
    async with get_db_connection() as db:
        stats = {}

        # Count stocks
        async with db.execute("SELECT COUNT(*) FROM stocks") as cursor:
            row = await cursor.fetchone()
            stats["stocks_count"] = row[0] if row else 0

        # Count watchlist items
        async with db.execute("SELECT COUNT(*) FROM watchlist") as cursor:
            row = await cursor.fetchone()
            stats["watchlist_count"] = row[0] if row else 0

        # Count analyses
        async with db.execute("SELECT COUNT(*) FROM analyses") as cursor:
            row = await cursor.fetchone()
            stats["analyses_count"] = row[0] if row else 0

        # Count performance checks
        async with db.execute("SELECT COUNT(*) FROM performance_checks") as cursor:
            row = await cursor.fetchone()
            stats["performance_checks_count"] = row[0] if row else 0

        return stats
