"""
Analysis API endpoints.

Provides endpoints for stock analysis operations.
"""

from fastapi import APIRouter, HTTPException, Query
from typing import Optional, List
from datetime import datetime
from pydantic import BaseModel

from ..database.dao import AnalysisDAO, StockDAO
from ..database.models import AnalysisResult, AnalysisCreate, APIResponse

router = APIRouter()


class AnalyzeRequest(BaseModel):
    """Request model for analyzing a stock."""
    analysis_style: str = "Conservative"


@router.post("/analyze/{ticker}", response_model=AnalysisResult)
async def analyze_stock(ticker: str, request: AnalyzeRequest):
    """
    Trigger analysis for a single stock.

    This endpoint:
    1. Fetches stock data via the Data Collector agent
    2. Runs LLM analysis via the Investment Analyst agent
    3. Generates charts and report
    4. Stores the analysis result
    """
    ticker = ticker.strip().upper()

    # Validate ticker
    if not ticker or len(ticker) > 10:
        raise HTTPException(status_code=400, detail="Invalid ticker symbol")

    # Validate analysis style
    if request.analysis_style not in ("Conservative", "Aggressive"):
        raise HTTPException(status_code=400, detail="Invalid analysis style")

    # TODO: Implement actual analysis using agents
    # For now, create a placeholder to demonstrate the API structure

    # This will be replaced with actual agent-based analysis
    raise HTTPException(
        status_code=501,
        detail="Analysis not yet implemented. Agents module pending."
    )


@router.get("/analysis/{ticker}", response_model=AnalysisResult)
async def get_latest_analysis(ticker: str):
    """Get the most recent analysis for a stock."""
    ticker = ticker.strip().upper()

    analysis = await AnalysisDAO.get_latest_by_ticker(ticker)
    if not analysis:
        raise HTTPException(
            status_code=404,
            detail=f"No analysis found for {ticker}"
        )

    return analysis


@router.get("/analysis/{ticker}/history", response_model=List[AnalysisResult])
async def get_analysis_history(
    ticker: str,
    limit: int = Query(default=10, ge=1, le=100),
    offset: int = Query(default=0, ge=0)
):
    """Get all past analyses for a stock."""
    ticker = ticker.strip().upper()

    analyses = await AnalysisDAO.get_history(
        ticker=ticker,
        limit=limit,
        offset=offset
    )

    return analyses


@router.get("/analysis/{analysis_id}/report")
async def get_analysis_report(analysis_id: int):
    """Get the markdown report for an analysis."""
    analysis = await AnalysisDAO.get_by_id(analysis_id)

    if not analysis:
        raise HTTPException(status_code=404, detail="Analysis not found")

    if not analysis.report_path:
        raise HTTPException(status_code=404, detail="No report available")

    # TODO: Return the actual markdown file content
    return APIResponse(
        success=True,
        data={"report_path": analysis.report_path}
    )


@router.get("/analyses", response_model=List[AnalysisResult])
async def get_all_analyses(
    ticker: Optional[str] = None,
    start_date: Optional[datetime] = None,
    end_date: Optional[datetime] = None,
    limit: int = Query(default=20, ge=1, le=100),
    offset: int = Query(default=0, ge=0)
):
    """Get all analyses with optional filters."""
    if ticker:
        ticker = ticker.strip().upper()

    return await AnalysisDAO.get_history(
        ticker=ticker,
        start_date=start_date,
        end_date=end_date,
        limit=limit,
        offset=offset
    )


@router.get("/analyses/count")
async def get_analyses_count(
    ticker: Optional[str] = None,
    start_date: Optional[datetime] = None,
    end_date: Optional[datetime] = None
):
    """Get the count of analyses matching filters."""
    if ticker:
        ticker = ticker.strip().upper()

    count = await AnalysisDAO.count(
        ticker=ticker,
        start_date=start_date,
        end_date=end_date
    )

    return {"count": count}


@router.get("/analyses/tickers")
async def get_analyzed_tickers():
    """Get list of all tickers that have been analyzed."""
    tickers = await AnalysisDAO.get_unique_tickers()
    return {"tickers": tickers}
