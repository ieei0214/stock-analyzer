"""
Stock Data API endpoints.

Provides endpoints for fetching and managing stock data.
"""

from fastapi import APIRouter, HTTPException
from typing import Optional

from ..database.dao import StockDAO, StockDataDAO
from ..database.models import Stock, StockCreate, APIResponse

router = APIRouter()


@router.get("/{ticker}", response_model=Stock)
async def get_stock(ticker: str):
    """Get stock overview data."""
    ticker = ticker.strip().upper()

    stock = await StockDAO.get_by_ticker(ticker)
    if not stock:
        raise HTTPException(
            status_code=404,
            detail=f"Stock {ticker} not found"
        )

    return stock


@router.get("/{ticker}/data")
async def get_stock_data(ticker: str, data_type: Optional[str] = None):
    """Get cached stock data."""
    ticker = ticker.strip().upper()

    if data_type:
        data = await StockDataDAO.get(ticker, data_type)
        if not data:
            raise HTTPException(
                status_code=404,
                detail=f"No cached data for {ticker}/{data_type}"
            )
        return {"ticker": ticker, "data_type": data_type, "data": data}

    # Return all data types
    data_types = ["price_history", "fundamentals", "technicals", "news"]
    result = {"ticker": ticker, "data": {}}

    for dt in data_types:
        cached = await StockDataDAO.get(ticker, dt)
        if cached:
            result["data"][dt] = cached

    return result


@router.post("/{ticker}/refresh", response_model=APIResponse)
async def refresh_stock_data(ticker: str):
    """Force refresh cached stock data."""
    ticker = ticker.strip().upper()

    # Invalidate all cached data for this ticker
    await StockDataDAO.invalidate(ticker)

    # TODO: Trigger data collector agent to fetch fresh data
    # For now, just invalidate the cache

    return APIResponse(
        success=True,
        message=f"Cache invalidated for {ticker}. Fresh data will be fetched on next analysis."
    )


@router.get("/{ticker}/charts")
async def get_stock_charts(ticker: str):
    """Get chart images for a stock."""
    ticker = ticker.strip().upper()

    # TODO: Implement chart generation and retrieval
    # For now, return placeholder

    return APIResponse(
        success=False,
        message="Chart generation not yet implemented"
    )


@router.get("/{ticker}/cache-status")
async def get_cache_status(ticker: str):
    """Get cache status for a stock's data."""
    ticker = ticker.strip().upper()

    data_types = ["price_history", "fundamentals", "technicals", "news"]
    status = {}

    for dt in data_types:
        is_cached = await StockDataDAO.is_cached(ticker, dt)
        status[dt] = "cached" if is_cached else "not_cached"

    return {"ticker": ticker, "cache_status": status}
