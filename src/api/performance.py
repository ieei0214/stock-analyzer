"""
Performance Tracking API endpoints.

Provides endpoints for tracking and analyzing recommendation performance.
"""

import logging
from fastapi import APIRouter, HTTPException, Query
from typing import List, Optional
from datetime import datetime

from ..database.dao import PerformanceDAO, AnalysisDAO
from ..database.models import PerformanceCheck, PerformanceStats, APIResponse
from ..agents.data_collector import DataCollectorAgent

router = APIRouter()
logger = logging.getLogger(__name__)


@router.post("/check/{analysis_id}", response_model=PerformanceCheck)
async def check_performance(analysis_id: int):
    """
    Trigger a performance check for a specific analysis.

    Fetches the current price and compares it to the price at analysis time.
    """
    # Get the analysis
    analysis = await AnalysisDAO.get_by_id(analysis_id)
    if not analysis:
        raise HTTPException(status_code=404, detail="Analysis not found")

    if not analysis.price_at_analysis:
        raise HTTPException(
            status_code=400,
            detail="Analysis does not have price data"
        )

    try:
        # Fetch current price using yfinance
        collector = DataCollectorAgent(analysis.ticker)
        price_data = collector.get_current_price()
        current_price = price_data.get("current_price")

        if not current_price:
            raise HTTPException(
                status_code=503,
                detail=f"Unable to fetch current price for {analysis.ticker}"
            )

        # Calculate gain/loss
        gain_loss_percent = ((current_price - analysis.price_at_analysis) / analysis.price_at_analysis) * 100

        # Calculate time period in days
        analysis_date = analysis.analysis_date
        if isinstance(analysis_date, str):
            analysis_date = datetime.fromisoformat(analysis_date.replace('Z', '+00:00'))
        time_period_days = (datetime.now() - analysis_date.replace(tzinfo=None)).days

        # Determine outcome based on recommendation and gain/loss
        # For Buy recommendations: positive gain is Profitable
        # For Sell recommendations: negative gain is Profitable
        # For Hold: neutral unless significant change
        outcome = "Neutral"
        if analysis.recommendation == "Buy":
            if gain_loss_percent > 1:
                outcome = "Profitable"
            elif gain_loss_percent < -1:
                outcome = "Loss"
        elif analysis.recommendation == "Sell":
            if gain_loss_percent < -1:
                outcome = "Profitable"  # Sell was correct
            elif gain_loss_percent > 1:
                outcome = "Loss"  # Sell was wrong
        else:  # Hold
            if abs(gain_loss_percent) <= 3:
                outcome = "Profitable"  # Hold was correct - price stable
            else:
                outcome = "Neutral"

        # Store the performance check
        performance_check = await PerformanceDAO.create(
            analysis_id=analysis_id,
            price_at_check=current_price,
            gain_loss_percent=round(gain_loss_percent, 2),
            time_period_days=time_period_days,
            outcome=outcome
        )

        logger.info(f"Performance check for analysis {analysis_id}: {gain_loss_percent:.2f}% ({outcome})")

        return performance_check

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Performance check failed for analysis {analysis_id}: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to check performance: {str(e)}"
        )


@router.get("/stats", response_model=PerformanceStats)
async def get_performance_stats():
    """Get overall performance statistics."""
    return await PerformanceDAO.get_stats()


@router.get("/history", response_model=List[PerformanceCheck])
async def get_performance_history(
    analysis_id: Optional[int] = None,
    limit: int = Query(default=50, ge=1, le=200),
    offset: int = Query(default=0, ge=0)
):
    """Get performance check history."""
    if analysis_id:
        return await PerformanceDAO.get_by_analysis(analysis_id)

    # TODO: Implement paginated history for all checks
    return []


@router.get("/analysis/{analysis_id}", response_model=List[PerformanceCheck])
async def get_analysis_performance(analysis_id: int):
    """Get all performance checks for a specific analysis."""
    # Verify analysis exists
    analysis = await AnalysisDAO.get_by_id(analysis_id)
    if not analysis:
        raise HTTPException(status_code=404, detail="Analysis not found")

    return await PerformanceDAO.get_by_analysis(analysis_id)
