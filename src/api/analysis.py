"""
Analysis API endpoints.

Provides endpoints for stock analysis operations.
"""

import os
import logging
from fastapi import APIRouter, HTTPException, Query
from typing import Optional, List
from datetime import datetime
from pydantic import BaseModel

from fastapi.responses import PlainTextResponse

from ..database.dao import AnalysisDAO, StockDAO, StockDataDAO, SettingsDAO
from ..database.models import AnalysisResult, StockCreate, APIResponse
from ..agents import DataCollectorAgent, InvestmentAnalystAgent, validate_stock_ticker
from ..reports.generator import report_generator

router = APIRouter()
logger = logging.getLogger(__name__)


class AnalyzeRequest(BaseModel):
    """Request model for analyzing a stock."""
    analysis_style: str = "Conservative"


def is_placeholder_key(key: str) -> bool:
    """Check if an API key looks like a placeholder."""
    if not key:
        return True
    placeholder_patterns = [
        "your-", "your_", "placeholder", "example", "test",
        "xxx", "sk-xxx", "enter-", "insert-", "add-your-"
    ]
    key_lower = key.lower()
    return any(pattern in key_lower for pattern in placeholder_patterns)


async def get_llm_settings():
    """Get LLM configuration from settings."""
    # Read settings from environment variables (same as settings.py does)
    provider = os.getenv("LLM_PROVIDER", "openai")

    # Get API key from environment (not stored in database for security)
    if provider == "openai":
        api_key = os.getenv("OPENAI_API_KEY")
    else:
        api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")

    # Check for placeholder keys
    if is_placeholder_key(api_key):
        api_key = None

    model = os.getenv("MODEL_NAME", "gpt-4" if provider == "openai" else "gemini-pro")

    return provider, api_key, model


@router.post("/analyze/{ticker}", response_model=AnalysisResult)
async def analyze_stock_endpoint(ticker: str, request: AnalyzeRequest):
    """
    Trigger analysis for a single stock.

    This endpoint:
    1. Fetches stock data via the Data Collector agent
    2. Runs LLM analysis via the Investment Analyst agent
    3. Stores the analysis result
    """
    ticker = ticker.strip().upper()

    # Validate ticker format
    if not ticker or len(ticker) > 10:
        raise HTTPException(status_code=400, detail="Invalid ticker symbol format")

    # Basic character validation
    if not ticker.replace(".", "").replace("-", "").isalnum():
        raise HTTPException(status_code=400, detail="Invalid ticker symbol characters")

    # Validate analysis style
    if request.analysis_style not in ("Conservative", "Aggressive"):
        raise HTTPException(status_code=400, detail="Invalid analysis style. Must be 'Conservative' or 'Aggressive'")

    # Get LLM settings
    provider, api_key, model = await get_llm_settings()

    if not api_key:
        raise HTTPException(
            status_code=400,
            detail=f"No API key configured for {provider}. Please configure your API key in Settings."
        )

    try:
        # Step 1: Collect stock data
        logger.info(f"Collecting data for {ticker}")
        collector = DataCollectorAgent(ticker)

        # Validate the ticker is real
        if not collector.validate_ticker():
            raise HTTPException(
                status_code=400,
                detail=f"Invalid ticker symbol: {ticker}. Could not find stock data."
            )

        stock_data = collector.collect_all_data()

        # Ensure stock exists in database
        company_info = stock_data.get("company", {})
        await StockDAO.create(StockCreate(
            ticker=ticker,
            company_name=company_info.get("company_name"),
            sector=company_info.get("sector"),
            industry=company_info.get("industry"),
        ))

        # Cache the stock data
        cache_hours = int((await SettingsDAO.get("cache_duration_hours")) or 1)
        await StockDataDAO.save(ticker, "full_data", stock_data, cache_hours)

        # Step 2: Run LLM analysis
        logger.info(f"Running {provider} analysis for {ticker}")
        analyst = InvestmentAnalystAgent(provider=provider, api_key=api_key, model=model)
        analysis_result = await analyst.analyze(stock_data, request.analysis_style)

        # Step 3: Generate report
        report_path = None
        try:
            analysis_data = {
                "recommendation": analysis_result.recommendation,
                "confidence": analysis_result.confidence,
                "analysis_style": request.analysis_style,
                "reasoning": analysis_result.reasoning,
                "price_at_analysis": analysis_result.price_at_analysis,
                "llm_provider": provider,
                "llm_model": model,
            }
            report_path = report_generator.generate_report(
                ticker=ticker,
                stock_data=stock_data,
                analysis=analysis_data,
                save_charts=True
            )
            logger.info(f"Generated report for {ticker}: {report_path}")
        except Exception as e:
            logger.warning(f"Failed to generate report for {ticker}: {e}")
            # Continue without report - not critical

        # Step 4: Store the analysis
        saved_analysis = await AnalysisDAO.create(
            ticker=ticker,
            recommendation=analysis_result.recommendation,
            confidence_level=analysis_result.confidence,
            analysis_style=request.analysis_style,
            reasoning=analysis_result.reasoning,
            price_at_analysis=analysis_result.price_at_analysis,
            llm_provider=provider,
            llm_model=model,
            report_path=report_path,
        )

        logger.info(f"Analysis complete for {ticker}: {analysis_result.recommendation}")

        return saved_analysis

    except HTTPException:
        raise
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Analysis failed for {ticker}: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Analysis failed: {str(e)}"
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

    # Return the actual markdown file content
    content = report_generator.get_report_content(analysis.report_path)
    if not content:
        raise HTTPException(status_code=404, detail="Report file not found")

    return PlainTextResponse(content=content, media_type="text/markdown")


@router.get("/analysis/{analysis_id}/report/download")
async def download_analysis_report(analysis_id: int):
    """Download the markdown report for an analysis as a file."""
    analysis = await AnalysisDAO.get_by_id(analysis_id)

    if not analysis:
        raise HTTPException(status_code=404, detail="Analysis not found")

    if not analysis.report_path:
        raise HTTPException(status_code=404, detail="No report available")

    content = report_generator.get_report_content(analysis.report_path)
    if not content:
        raise HTTPException(status_code=404, detail="Report file not found")

    # Get filename from path
    filename = os.path.basename(analysis.report_path)

    return PlainTextResponse(
        content=content,
        media_type="text/markdown",
        headers={
            "Content-Disposition": f"attachment; filename={analysis.ticker}_{filename}"
        }
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


@router.get("/analyses/export")
async def export_analyses_csv(
    ticker: Optional[str] = None,
    start_date: Optional[datetime] = None,
    end_date: Optional[datetime] = None
):
    """Export analyses to CSV format."""
    import csv
    import io

    if ticker:
        ticker = ticker.strip().upper()

    # Get all analyses matching the filters (no limit for export)
    analyses = await AnalysisDAO.get_history(
        ticker=ticker,
        start_date=start_date,
        end_date=end_date,
        limit=1000  # Reasonable max for export
    )

    if not analyses:
        raise HTTPException(status_code=404, detail="No analyses found for export")

    # Create CSV in memory
    output = io.StringIO()
    writer = csv.writer(output)

    # Write header
    writer.writerow([
        "Date",
        "Ticker",
        "Recommendation",
        "Confidence",
        "Analysis Style",
        "Price at Analysis",
        "Current Price",
        "Gain/Loss %",
        "LLM Provider",
        "LLM Model",
        "Reasoning"
    ])

    # Write data rows
    for analysis in analyses:
        writer.writerow([
            analysis.analysis_date.strftime("%Y-%m-%d %H:%M") if analysis.analysis_date else "",
            analysis.ticker,
            analysis.recommendation,
            analysis.confidence_level,
            analysis.analysis_style,
            f"{analysis.price_at_analysis:.2f}" if analysis.price_at_analysis else "",
            f"{analysis.current_price:.2f}" if hasattr(analysis, 'current_price') and analysis.current_price else "",
            f"{analysis.gain_loss_percent:.2f}" if hasattr(analysis, 'gain_loss_percent') and analysis.gain_loss_percent is not None else "",
            analysis.llm_provider or "",
            analysis.llm_model or "",
            (analysis.reasoning or "").replace("\n", " ").replace("\r", "")[:500]  # Truncate long reasoning
        ])

    csv_content = output.getvalue()
    output.close()

    # Return as downloadable file
    filename = f"stock_analyses_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"

    return PlainTextResponse(
        content=csv_content,
        media_type="text/csv",
        headers={
            "Content-Disposition": f"attachment; filename={filename}"
        }
    )
