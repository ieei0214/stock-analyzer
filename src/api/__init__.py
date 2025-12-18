# Stock Analyzer - API Package
"""
FastAPI route handlers for the web API.

Routers:
- watchlist: Watchlist management endpoints
- analysis: Stock analysis endpoints
- stock: Stock data endpoints
- performance: Performance tracking endpoints
- settings: Application settings endpoints
- tasks: Background task management endpoints
"""

from . import watchlist
from . import analysis
from . import stock
from . import performance
from . import settings
from . import tasks

__all__ = [
    "watchlist",
    "analysis",
    "stock",
    "performance",
    "settings",
    "tasks"
]
