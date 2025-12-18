"""
Stock Analyzer Agents.

This package contains the two main agents:
- DataCollectorAgent: Fetches stock data from yfinance
- InvestmentAnalystAgent: LLM-powered analysis and recommendations
"""

from .data_collector import (
    DataCollectorAgent,
    collect_stock_data,
    validate_stock_ticker,
)
from .analyst import (
    InvestmentAnalystAgent,
    AnalysisResult,
    analyze_stock,
)

__all__ = [
    "DataCollectorAgent",
    "collect_stock_data",
    "validate_stock_ticker",
    "InvestmentAnalystAgent",
    "AnalysisResult",
    "analyze_stock",
]
