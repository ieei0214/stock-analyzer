"""
Watchlist API endpoints.

Provides endpoints for managing the stock watchlist.
"""

from fastapi import APIRouter, HTTPException, BackgroundTasks
from typing import List
from pydantic import BaseModel

from ..database.dao import WatchlistDAO, StockDAO
from ..database.models import WatchlistItem, WatchlistItemCreate, APIResponse

router = APIRouter()


class AddStockRequest(BaseModel):
    """Request model for adding a stock."""
    ticker: str


@router.get("", response_model=List[WatchlistItem])
async def get_watchlist():
    """Get all stocks in the watchlist."""
    return await WatchlistDAO.get_all()


@router.post("", response_model=WatchlistItem)
async def add_to_watchlist(request: AddStockRequest):
    """Add a stock to the watchlist."""
    ticker = request.ticker.strip().upper()

    # Validate ticker format
    if not ticker or len(ticker) > 10:
        raise HTTPException(status_code=400, detail="Invalid ticker symbol")

    # Check if already in watchlist
    if await WatchlistDAO.exists(ticker):
        raise HTTPException(
            status_code=400,
            detail=f"{ticker} is already in your watchlist"
        )

    try:
        item = await WatchlistDAO.add(WatchlistItemCreate(ticker=ticker))
        return item
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/{ticker}", response_model=APIResponse)
async def remove_from_watchlist(ticker: str):
    """Remove a stock from the watchlist."""
    ticker = ticker.strip().upper()

    if not await WatchlistDAO.exists(ticker):
        raise HTTPException(
            status_code=404,
            detail=f"{ticker} not found in watchlist"
        )

    removed = await WatchlistDAO.remove(ticker)
    if removed:
        return APIResponse(success=True, message=f"{ticker} removed from watchlist")
    else:
        raise HTTPException(status_code=500, detail="Failed to remove stock")


@router.post("/analyze-all", response_model=APIResponse)
async def analyze_all_watchlist(background_tasks: BackgroundTasks):
    """Trigger analysis for all stocks in the watchlist."""
    items = await WatchlistDAO.get_all()

    if not items:
        raise HTTPException(status_code=400, detail="Watchlist is empty")

    # TODO: Implement batch analysis with background tasks
    # For now, return a placeholder response
    tickers = [item.ticker for item in items]

    return APIResponse(
        success=True,
        message=f"Started analysis for {len(tickers)} stocks",
        data={"tickers": tickers, "task_id": "placeholder"}
    )


@router.get("/count")
async def get_watchlist_count():
    """Get the number of stocks in the watchlist."""
    count = await WatchlistDAO.count()
    return {"count": count}
