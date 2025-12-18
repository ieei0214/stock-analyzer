"""
Performance Tracking API endpoints.

Provides endpoints for tracking and analyzing recommendation performance.
"""

from fastapi import APIRouter, HTTPException, Query
from typing import List, Optional
from datetime import datetime

from ..database.dao import PerformanceDAO, AnalysisDAO
from ..database.models import PerformanceCheck, PerformanceStats, APIResponse

router = APIRouter()


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

    # TODO: Fetch current price using yfinance
    # For now, return placeholder
    raise HTTPException(
        status_code=501,
        detail="Performance check not yet implemented. Data collector pending."
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
