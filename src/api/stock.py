"""
Stock Data API endpoints.

Provides endpoints for fetching and managing stock data.
"""

import logging
from fastapi import APIRouter, HTTPException
from fastapi.responses import JSONResponse
from typing import Optional

from ..database.dao import StockDAO, StockDataDAO, SettingsDAO
from ..database.models import Stock, StockCreate, APIResponse
from ..agents import DataCollectorAgent, validate_stock_ticker
from ..charts.generator import chart_generator

router = APIRouter()
logger = logging.getLogger(__name__)


@router.get("/{ticker}", response_model=Stock)
async def get_stock(ticker: str):
    """Get stock overview data."""
    ticker = ticker.strip().upper()

    stock = await StockDAO.get_by_ticker(ticker)
    if not stock:
        # Try to fetch from yfinance and create
        try:
            collector = DataCollectorAgent(ticker)
            if collector.validate_ticker():
                company_info = collector.get_company_info()
                stock = await StockDAO.create(StockCreate(
                    ticker=ticker,
                    company_name=company_info.get("company_name"),
                    sector=company_info.get("sector"),
                    industry=company_info.get("industry"),
                ))
            else:
                raise HTTPException(
                    status_code=404,
                    detail=f"Stock {ticker} not found"
                )
        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Error fetching stock {ticker}: {e}")
            raise HTTPException(
                status_code=404,
                detail=f"Stock {ticker} not found"
            )

    return stock


@router.get("/{ticker}/price")
async def get_stock_price(ticker: str):
    """Get current price data for a stock."""
    ticker = ticker.strip().upper()

    try:
        collector = DataCollectorAgent(ticker)
        if not collector.validate_ticker():
            raise HTTPException(
                status_code=404,
                detail=f"Invalid ticker symbol: {ticker}"
            )

        price_data = collector.get_current_price()
        return {"ticker": ticker, "price": price_data}

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error fetching price for {ticker}: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to fetch price data: {str(e)}"
        )


@router.get("/{ticker}/data")
async def get_stock_data(ticker: str, data_type: Optional[str] = None):
    """Get cached stock data or fetch fresh if not cached."""
    ticker = ticker.strip().upper()

    if data_type:
        # Try cache first
        data = await StockDataDAO.get(ticker, data_type)
        if data:
            return {"ticker": ticker, "data_type": data_type, "data": data, "from_cache": True}

        # Fetch fresh data
        try:
            collector = DataCollectorAgent(ticker)
            if not collector.validate_ticker():
                raise HTTPException(status_code=404, detail=f"Invalid ticker: {ticker}")

            fresh_data = None
            if data_type == "price_history":
                fresh_data = collector.get_price_history()
            elif data_type == "fundamentals":
                fresh_data = collector.get_financial_metrics()
            elif data_type == "technicals":
                fresh_data = collector.calculate_technical_indicators()
            elif data_type == "news":
                fresh_data = collector.get_news()
            elif data_type == "analyst":
                fresh_data = collector.get_analyst_ratings()
            elif data_type == "insider":
                fresh_data = collector.get_insider_trading()
            else:
                raise HTTPException(status_code=400, detail=f"Unknown data type: {data_type}")

            # Cache the data
            cache_hours = int((await SettingsDAO.get("cache_duration_hours")) or 1)
            await StockDataDAO.save(ticker, data_type, fresh_data, cache_hours)

            return {"ticker": ticker, "data_type": data_type, "data": fresh_data, "from_cache": False}

        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Error fetching {data_type} for {ticker}: {e}")
            raise HTTPException(status_code=500, detail=str(e))

    # Return all cached data types
    data_types = ["price_history", "fundamentals", "technicals", "news", "analyst", "insider"]
    result = {"ticker": ticker, "data": {}}

    for dt in data_types:
        cached = await StockDataDAO.get(ticker, dt)
        if cached:
            result["data"][dt] = cached

    return result


@router.post("/{ticker}/refresh", response_model=APIResponse)
async def refresh_stock_data(ticker: str):
    """Force refresh cached stock data by fetching fresh data from yfinance."""
    ticker = ticker.strip().upper()

    try:
        # Validate ticker
        collector = DataCollectorAgent(ticker)
        if not collector.validate_ticker():
            raise HTTPException(status_code=404, detail=f"Invalid ticker: {ticker}")

        # Invalidate all cached data for this ticker
        await StockDataDAO.invalidate(ticker)

        # Fetch and cache fresh data
        cache_hours = int((await SettingsDAO.get("cache_duration_hours")) or 1)

        stock_data = collector.collect_all_data()

        # Cache individual components
        await StockDataDAO.save(ticker, "full_data", stock_data, cache_hours)
        await StockDataDAO.save(ticker, "price_history", stock_data.get("history", {}), cache_hours)
        await StockDataDAO.save(ticker, "fundamentals", stock_data.get("financials", {}), cache_hours)
        await StockDataDAO.save(ticker, "technicals", stock_data.get("technicals", {}), cache_hours)
        await StockDataDAO.save(ticker, "news", stock_data.get("news", []), cache_hours)
        await StockDataDAO.save(ticker, "analyst", stock_data.get("analyst", {}), cache_hours)
        await StockDataDAO.save(ticker, "insider", stock_data.get("insider_trading", []), cache_hours)

        # Update stock info in database
        company_info = stock_data.get("company", {})
        await StockDAO.create(StockCreate(
            ticker=ticker,
            company_name=company_info.get("company_name"),
            sector=company_info.get("sector"),
            industry=company_info.get("industry"),
        ))

        return APIResponse(
            success=True,
            message=f"Successfully refreshed data for {ticker}"
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error refreshing data for {ticker}: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to refresh data: {str(e)}"
        )


@router.get("/{ticker}/charts")
async def get_stock_charts(ticker: str):
    """Get chart data for a stock."""
    ticker = ticker.strip().upper()

    try:
        collector = DataCollectorAgent(ticker)
        if not collector.validate_ticker():
            raise HTTPException(status_code=404, detail=f"Invalid ticker: {ticker}")

        # Get data for charts
        history = collector.get_price_history()
        technicals = collector.calculate_technical_indicators()

        return {
            "ticker": ticker,
            "price_chart": {
                "dates": history.get("dates", [])[-100:],
                "close": history.get("close", [])[-100:],
                "volume": history.get("volume", [])[-100:],
            },
            "moving_averages": {
                "ma_50": technicals.get("ma_50_history", []),
                "ma_200": technicals.get("ma_200_history", []),
            },
            "rsi": {
                "values": technicals.get("rsi_history", []),
            },
            "macd": {
                "macd": technicals.get("macd_history", []),
                "signal": technicals.get("macd_signal_history", []),
            },
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting charts for {ticker}: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to generate charts: {str(e)}"
        )


@router.get("/{ticker}/cache-status")
async def get_cache_status(ticker: str):
    """Get cache status for a stock's data."""
    ticker = ticker.strip().upper()

    data_types = ["full_data", "price_history", "fundamentals", "technicals", "news", "analyst"]
    status = {}

    for dt in data_types:
        is_cached = await StockDataDAO.is_cached(ticker, dt)
        status[dt] = "cached" if is_cached else "not_cached"

    return {"ticker": ticker, "cache_status": status}


@router.get("/{ticker}/validate")
async def validate_ticker(ticker: str):
    """Validate if a ticker symbol is valid."""
    ticker = ticker.strip().upper()

    try:
        is_valid = validate_stock_ticker(ticker)
        return {
            "ticker": ticker,
            "valid": is_valid,
            "message": f"{ticker} is a valid ticker" if is_valid else f"{ticker} is not a valid ticker"
        }
    except Exception as e:
        return {
            "ticker": ticker,
            "valid": False,
            "message": str(e)
        }


@router.get("/{ticker}/chart-images")
async def get_chart_images(ticker: str, chart_type: Optional[str] = None):
    """Get chart images as base64-encoded data URLs.

    Args:
        ticker: Stock ticker symbol
        chart_type: Optional specific chart type (price, volume, rsi, macd)
                   If not specified, returns all charts

    Returns:
        Dictionary of chart names to base64 data URLs
    """
    ticker = ticker.strip().upper()

    try:
        collector = DataCollectorAgent(ticker)
        if not collector.validate_ticker():
            raise HTTPException(status_code=404, detail=f"Invalid ticker: {ticker}")

        # Get data for charts
        history = collector.get_price_history()
        technicals = collector.calculate_technical_indicators()

        # Prepare price data format for chart generator
        price_data = {
            "dates": history.get("dates", []),
            "open": history.get("open", []),
            "close": history.get("close", []),
            "volume": history.get("volume", []),
        }

        if chart_type:
            # Generate specific chart
            chart_type = chart_type.lower()
            if chart_type == "price":
                chart = chart_generator.generate_price_chart(
                    price_data, technicals, ticker)
            elif chart_type == "volume":
                chart = chart_generator.generate_volume_chart(
                    price_data, ticker)
            elif chart_type == "rsi":
                chart = chart_generator.generate_rsi_chart(
                    technicals, history.get("dates"), ticker)
            elif chart_type == "macd":
                chart = chart_generator.generate_macd_chart(
                    technicals, history.get("dates"), ticker)
            else:
                raise HTTPException(
                    status_code=400,
                    detail=f"Unknown chart type: {chart_type}. Valid types: price, volume, rsi, macd"
                )
            return {"ticker": ticker, "chart_type": chart_type, "image": chart}

        # Generate all charts
        charts = chart_generator.generate_all_charts(
            ticker, price_data, technicals)

        return {
            "ticker": ticker,
            "charts": charts
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error generating chart images for {ticker}: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to generate chart images: {str(e)}"
        )
