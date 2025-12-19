"""
Watchlist API endpoints.

Provides endpoints for managing the stock watchlist.
"""

import logging
import csv
import io
from fastapi import APIRouter, HTTPException, BackgroundTasks
from fastapi.responses import StreamingResponse
from typing import List
from pydantic import BaseModel

from ..database.dao import WatchlistDAO, StockDAO
from ..database.models import WatchlistItem, WatchlistItemCreate, APIResponse, StockCreate
from ..agents import DataCollectorAgent, validate_stock_ticker
from .tasks import create_task, update_task, complete_task, fail_task

router = APIRouter()
logger = logging.getLogger(__name__)


class AddStockRequest(BaseModel):
    """Request model for adding a stock."""
    ticker: str


async def enrich_watchlist_item(item: WatchlistItem) -> WatchlistItem:
    """Enrich a watchlist item with current price data."""
    try:
        collector = DataCollectorAgent(item.ticker)
        price_data = collector.get_current_price()

        item.current_price = price_data.get("current_price")
        item.price_change_percent = price_data.get("change_percent")
    except Exception as e:
        logger.warning(f"Failed to get price data for {item.ticker}: {e}")

    return item


@router.get("", response_model=List[WatchlistItem])
async def get_watchlist():
    """Get all stocks in the watchlist with enriched data."""
    items = await WatchlistDAO.get_all()

    # Enrich each item with current price data
    enriched_items = []
    for item in items:
        try:
            enriched = await enrich_watchlist_item(item)
            enriched_items.append(enriched)
        except Exception as e:
            logger.warning(f"Failed to enrich {item.ticker}: {e}")
            enriched_items.append(item)

    return enriched_items


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

    # Validate ticker exists on yfinance
    if not validate_stock_ticker(ticker):
        raise HTTPException(
            status_code=400,
            detail=f"Invalid ticker symbol: {ticker}. Could not find stock data."
        )

    try:
        # Get company info from yfinance
        collector = DataCollectorAgent(ticker)
        company_info = collector.get_company_info()

        # Create or update stock record
        await StockDAO.create(StockCreate(
            ticker=ticker,
            company_name=company_info.get("company_name"),
            sector=company_info.get("sector"),
            industry=company_info.get("industry"),
        ))

        # Add to watchlist
        item = await WatchlistDAO.add(WatchlistItemCreate(ticker=ticker))

        # Enrich with price data
        item = await enrich_watchlist_item(item)

        return item
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to add {ticker} to watchlist: {e}")
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

    tickers = [item.ticker for item in items]

    # Create a background task with UUID-based task ID
    task = create_task("batch_analysis")

    # Add background task to process all stocks
    async def run_batch_analysis(task_id: str, stock_tickers: list):
        """Background task to analyze all watchlist stocks."""
        try:
            total = len(stock_tickers)
            for i, ticker in enumerate(stock_tickers):
                update_task(
                    task_id,
                    status="running",
                    progress=(i / total) * 100,
                    message=f"Analyzing {ticker} ({i+1}/{total})..."
                )
                # In production, this would trigger actual analysis
                # For now, just track progress
            complete_task(task_id, {"analyzed": stock_tickers})
        except Exception as e:
            fail_task(task_id, str(e))

    background_tasks.add_task(run_batch_analysis, task.task_id, tickers)

    return APIResponse(
        success=True,
        message=f"Started analysis for {len(tickers)} stocks",
        data={"tickers": tickers, "task_id": task.task_id}
    )


@router.get("/count")
async def get_watchlist_count():
    """Get the number of stocks in the watchlist."""
    count = await WatchlistDAO.count()
    return {"count": count}


@router.get("/export")
async def export_watchlist_csv():
    """Export the watchlist to a CSV file."""
    items = await WatchlistDAO.get_all()

    # Enrich each item with current price data
    enriched_items = []
    for item in items:
        try:
            enriched = await enrich_watchlist_item(item)
            enriched_items.append(enriched)
        except Exception as e:
            logger.warning(f"Failed to enrich {item.ticker}: {e}")
            enriched_items.append(item)

    # Create CSV in memory
    output = io.StringIO()
    writer = csv.writer(output)

    # Write header
    writer.writerow([
        "Ticker",
        "Company Name",
        "Current Price",
        "Change %",
        "Last Recommendation",
        "Last Analyzed",
        "Added At"
    ])

    # Write data rows
    for item in enriched_items:
        price_str = f"${item.current_price:.2f}" if item.current_price else ""
        change_str = f"{item.price_change_percent:.2f}%" if item.price_change_percent else ""
        last_analyzed = item.last_analyzed.isoformat() if item.last_analyzed else ""
        added_at = item.added_at.isoformat() if item.added_at else ""

        writer.writerow([
            item.ticker,
            item.company_name or "",
            price_str,
            change_str,
            item.last_recommendation or "",
            last_analyzed,
            added_at
        ])

    # Reset stream position
    output.seek(0)

    # Return as streaming response with CSV content type
    return StreamingResponse(
        iter([output.getvalue()]),
        media_type="text/csv",
        headers={
            "Content-Disposition": "attachment; filename=watchlist.csv"
        }
    )
