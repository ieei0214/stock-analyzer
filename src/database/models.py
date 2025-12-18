"""
Pydantic models for data validation and serialization.

These models are used for:
- Request/response validation in the API
- Data transfer between components
- Database record serialization
"""

from pydantic import BaseModel, Field, field_validator
from typing import Optional, List, Literal
from datetime import datetime
import re


# ===========================================
# Stock Models
# ===========================================

class StockBase(BaseModel):
    """Base model for stock data."""
    ticker: str = Field(..., min_length=1, max_length=10, description="Stock ticker symbol")

    @field_validator('ticker')
    @classmethod
    def validate_ticker(cls, v: str) -> str:
        """Validate and normalize ticker symbol."""
        v = v.strip().upper()
        # Only allow alphanumeric characters and common ticker symbols
        if not re.match(r'^[A-Z0-9.\-]+$', v):
            raise ValueError('Invalid ticker symbol format')
        return v


class StockCreate(StockBase):
    """Model for creating a new stock."""
    company_name: Optional[str] = None
    sector: Optional[str] = None
    industry: Optional[str] = None


class Stock(StockBase):
    """Full stock model with all fields."""
    company_name: Optional[str] = None
    sector: Optional[str] = None
    industry: Optional[str] = None
    created_at: Optional[datetime] = None
    last_updated: Optional[datetime] = None

    class Config:
        from_attributes = True


# ===========================================
# Watchlist Models
# ===========================================

class WatchlistItemCreate(BaseModel):
    """Model for adding a stock to watchlist."""
    ticker: str = Field(..., min_length=1, max_length=10)

    @field_validator('ticker')
    @classmethod
    def validate_ticker(cls, v: str) -> str:
        v = v.strip().upper()
        if not re.match(r'^[A-Z0-9.\-]+$', v):
            raise ValueError('Invalid ticker symbol format')
        return v


class WatchlistItem(BaseModel):
    """Full watchlist item model."""
    id: int
    ticker: str
    added_at: datetime
    company_name: Optional[str] = None
    current_price: Optional[float] = None
    price_change_percent: Optional[float] = None
    last_recommendation: Optional[str] = None
    last_analyzed: Optional[datetime] = None

    class Config:
        from_attributes = True


# ===========================================
# Analysis Models
# ===========================================

class AnalysisCreate(BaseModel):
    """Model for creating a new analysis."""
    ticker: str
    analysis_style: Literal['Conservative', 'Aggressive'] = 'Conservative'


class AnalysisResult(BaseModel):
    """Model for analysis result."""
    id: int
    ticker: str
    analysis_date: datetime
    recommendation: Literal['Buy', 'Sell', 'Hold']
    confidence_level: Literal['High', 'Medium', 'Low']
    reasoning: Optional[str] = None
    analysis_style: Literal['Conservative', 'Aggressive']
    price_at_analysis: Optional[float] = None
    report_path: Optional[str] = None
    llm_provider: Optional[str] = None
    llm_model: Optional[str] = None

    class Config:
        from_attributes = True


class AnalysisSummary(BaseModel):
    """Summary model for analysis list views."""
    id: int
    ticker: str
    analysis_date: datetime
    recommendation: str
    confidence_level: str
    price_at_analysis: Optional[float] = None
    current_price: Optional[float] = None
    gain_loss_percent: Optional[float] = None

    class Config:
        from_attributes = True


# ===========================================
# Performance Models
# ===========================================

class PerformanceCheck(BaseModel):
    """Model for performance check."""
    id: int
    analysis_id: int
    check_date: datetime
    price_at_check: float
    gain_loss_percent: Optional[float] = None
    time_period_days: Optional[int] = None
    outcome: Optional[Literal['Profitable', 'Loss', 'Neutral']] = None

    class Config:
        from_attributes = True


class PerformanceStats(BaseModel):
    """Aggregated performance statistics."""
    total_analyses: int = 0
    total_checked: int = 0
    profitable_count: int = 0
    loss_count: int = 0
    neutral_count: int = 0
    success_rate: float = 0.0
    average_gain_loss: float = 0.0


# ===========================================
# News Models
# ===========================================

class NewsItem(BaseModel):
    """Model for news item."""
    id: int
    ticker: str
    headline: str
    source: Optional[str] = None
    published_date: Optional[datetime] = None
    sentiment: Optional[Literal['Positive', 'Negative', 'Neutral']] = None
    url: Optional[str] = None
    fetched_at: Optional[datetime] = None

    class Config:
        from_attributes = True


# ===========================================
# Stock Data Models
# ===========================================

class StockDataCache(BaseModel):
    """Model for cached stock data."""
    id: int
    ticker: str
    data_type: str
    data_json: str
    fetched_at: datetime
    cache_expires_at: Optional[datetime] = None

    class Config:
        from_attributes = True


class PriceHistory(BaseModel):
    """Model for price history data."""
    dates: List[str]
    open: List[float]
    high: List[float]
    low: List[float]
    close: List[float]
    volume: List[int]


class FinancialMetrics(BaseModel):
    """Model for financial metrics."""
    pe_ratio: Optional[float] = None
    eps: Optional[float] = None
    revenue_growth: Optional[float] = None
    dividend_yield: Optional[float] = None
    debt_to_equity: Optional[float] = None
    profit_margin: Optional[float] = None
    market_cap: Optional[float] = None
    fifty_two_week_high: Optional[float] = None
    fifty_two_week_low: Optional[float] = None


class TechnicalIndicators(BaseModel):
    """Model for technical indicators."""
    ma_50: Optional[float] = None
    ma_200: Optional[float] = None
    rsi: Optional[float] = None
    macd: Optional[float] = None
    macd_signal: Optional[float] = None
    macd_histogram: Optional[float] = None


class AnalystRatings(BaseModel):
    """Model for analyst ratings."""
    strong_buy: int = 0
    buy: int = 0
    hold: int = 0
    sell: int = 0
    strong_sell: int = 0
    target_low: Optional[float] = None
    target_high: Optional[float] = None
    target_mean: Optional[float] = None


# ===========================================
# Settings Models
# ===========================================

class SettingsUpdate(BaseModel):
    """Model for updating settings."""
    llm_provider: Optional[Literal['openai', 'gemini']] = None
    openai_api_key: Optional[str] = None
    gemini_api_key: Optional[str] = None
    model_name: Optional[str] = None
    default_analysis_style: Optional[Literal['Conservative', 'Aggressive']] = None
    cache_duration_hours: Optional[int] = Field(None, ge=1, le=24)


class Settings(BaseModel):
    """Model for application settings."""
    llm_provider: str = 'openai'
    openai_api_key_set: bool = False
    gemini_api_key_set: bool = False
    openai_api_key_masked: Optional[str] = None  # Masked version for display
    gemini_api_key_masked: Optional[str] = None  # Masked version for display
    model_name: str = 'gpt-4'
    default_analysis_style: str = 'Conservative'
    cache_duration_hours: int = 1


# ===========================================
# Background Task Models
# ===========================================

class TaskStatus(BaseModel):
    """Model for background task status."""
    task_id: str
    status: Literal['pending', 'running', 'completed', 'failed', 'cancelled']
    progress: Optional[float] = None
    message: Optional[str] = None
    result: Optional[dict] = None
    created_at: datetime
    updated_at: Optional[datetime] = None


# ===========================================
# API Response Models
# ===========================================

class APIResponse(BaseModel):
    """Generic API response model."""
    success: bool
    message: Optional[str] = None
    data: Optional[dict] = None


class PaginatedResponse(BaseModel):
    """Paginated response model."""
    items: List
    total: int
    page: int
    page_size: int
    total_pages: int
