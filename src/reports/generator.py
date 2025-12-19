# Stock Analyzer - Report Generator
"""
Generate markdown analysis reports with embedded charts.

Reports include:
- Stock summary and key metrics
- Technical indicators
- News headlines
- LLM analysis and recommendation
- Embedded chart images
"""

import os
import logging
from datetime import datetime
from typing import Dict, Any, Optional
from ..charts.generator import chart_generator

logger = logging.getLogger(__name__)


class ReportGenerator:
    """Generate markdown analysis reports."""

    def __init__(self, reports_dir: str = "reports"):
        """Initialize report generator.

        Args:
            reports_dir: Base directory for storing reports
        """
        self.reports_dir = reports_dir
        os.makedirs(reports_dir, exist_ok=True)

    def _ensure_ticker_dir(self, ticker: str) -> str:
        """Ensure directory exists for ticker reports."""
        ticker_dir = os.path.join(self.reports_dir, ticker.upper())
        os.makedirs(ticker_dir, exist_ok=True)
        return ticker_dir

    def _format_number(self, value: Any, prefix: str = "", suffix: str = "") -> str:
        """Format a number for display."""
        if value is None:
            return "N/A"
        try:
            num = float(value)
            if abs(num) >= 1e12:
                return f"{prefix}{num/1e12:.2f}T{suffix}"
            elif abs(num) >= 1e9:
                return f"{prefix}{num/1e9:.2f}B{suffix}"
            elif abs(num) >= 1e6:
                return f"{prefix}{num/1e6:.2f}M{suffix}"
            elif abs(num) >= 1e3:
                return f"{prefix}{num/1e3:.2f}K{suffix}"
            else:
                return f"{prefix}{num:.2f}{suffix}"
        except (ValueError, TypeError):
            return str(value) if value else "N/A"

    def _format_percent(self, value: Any) -> str:
        """Format a percentage value."""
        if value is None:
            return "N/A"
        try:
            return f"{float(value):.2f}%"
        except (ValueError, TypeError):
            return "N/A"

    def _format_price(self, value: Any) -> str:
        """Format a price value."""
        if value is None:
            return "N/A"
        try:
            return f"${float(value):.2f}"
        except (ValueError, TypeError):
            return "N/A"

    def generate_report(
        self,
        ticker: str,
        stock_data: Dict[str, Any],
        analysis: Dict[str, Any],
        save_charts: bool = True
    ) -> str:
        """Generate a comprehensive markdown analysis report.

        Args:
            ticker: Stock ticker symbol
            stock_data: Complete stock data from DataCollector
            analysis: Analysis result from InvestmentAnalyst
            save_charts: Whether to save charts to files (True) or embed base64

        Returns:
            Path to the generated report file
        """
        ticker = ticker.upper()
        ticker_dir = self._ensure_ticker_dir(ticker)

        # Generate report filename
        date_str = datetime.now().strftime("%Y-%m-%d")
        report_filename = f"{date_str}_analysis.md"
        report_path = os.path.join(ticker_dir, report_filename)

        # Extract data sections
        company = stock_data.get("company", {})
        current_price = stock_data.get("current_price", {})
        financials = stock_data.get("financials", {})
        technicals = stock_data.get("technicals", {})
        analyst_ratings = stock_data.get("analyst_ratings", {})
        news = stock_data.get("news", [])
        insider = stock_data.get("insider", [])
        price_history = stock_data.get("price_history", {})

        # Generate charts (save to files)
        charts = {}
        if price_history:
            try:
                charts = chart_generator.generate_all_charts(
                    ticker, price_history, technicals, save_to_file=save_charts
                )
            except Exception as e:
                logger.error(f"Failed to generate charts for {ticker}: {e}")

        # Build the markdown report
        md_content = self._build_markdown(
            ticker=ticker,
            company=company,
            current_price=current_price,
            financials=financials,
            technicals=technicals,
            analyst_ratings=analyst_ratings,
            news=news,
            insider=insider,
            analysis=analysis,
            charts=charts,
            save_charts=save_charts
        )

        # Write the report file
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(md_content)

        logger.info(f"Generated report: {report_path}")
        return report_path

    def _build_markdown(
        self,
        ticker: str,
        company: Dict[str, Any],
        current_price: Dict[str, Any],
        financials: Dict[str, Any],
        technicals: Dict[str, Any],
        analyst_ratings: Dict[str, Any],
        news: list,
        insider: list,
        analysis: Dict[str, Any],
        charts: Dict[str, str],
        save_charts: bool = True
    ) -> str:
        """Build the markdown content for the report."""

        lines = []

        # Header
        lines.append(f"# {ticker} Stock Analysis Report")
        lines.append("")
        lines.append(f"**Generated:** {datetime.now().strftime('%B %d, %Y at %I:%M %p')}")
        lines.append(f"**Company:** {company.get('company_name', ticker)}")
        lines.append(f"**Sector:** {company.get('sector', 'N/A')}")
        lines.append(f"**Industry:** {company.get('industry', 'N/A')}")
        lines.append("")

        # Recommendation Banner
        recommendation = analysis.get("recommendation", "N/A")
        confidence = analysis.get("confidence", "N/A")

        rec_emoji = {"Buy": "&#128994;", "Sell": "&#128308;", "Hold": "&#128993;"}.get(recommendation, "&#9898;")

        lines.append("---")
        lines.append("")
        lines.append(f"## {rec_emoji} Recommendation: **{recommendation}**")
        lines.append(f"**Confidence Level:** {confidence}")
        lines.append(f"**Analysis Style:** {analysis.get('analysis_style', 'Conservative')}")
        lines.append(f"**Price at Analysis:** {self._format_price(analysis.get('price_at_analysis'))}")
        lines.append("")

        # Reasoning
        reasoning = analysis.get("reasoning", "")
        if reasoning:
            lines.append("### Analysis Reasoning")
            lines.append("")
            
            # Try to parse if it's JSON format
            import json
            import re
            reasoning_text = reasoning
            
            # Strip markdown code block markers if present
            reasoning_text = reasoning_text.replace('```json', '').replace('```', '').strip()
            
            # Check if it's JSON
            if reasoning_text.strip().startswith('{'):
                try:
                    parsed = json.loads(reasoning_text)
                    # Extract just the reasoning field if it exists
                    if 'reasoning' in parsed:
                        reasoning_text = parsed['reasoning']
                except json.JSONDecodeError as e:
                    # Try to extract partial reasoning from incomplete JSON
                    match = re.search(r'"reasoning":\s*"([^"]+)', reasoning_text)
                    if match:
                        reasoning_text = match.group(1) + "... (incomplete)"
                    else:
                        # Can't parse, show error message
                        lines.append("*Analysis reasoning data is incomplete or malformed.*")
                        lines.append("")
                        reasoning_text = None
            
            # Handle both string and list reasoning
            if reasoning_text:
                if isinstance(reasoning_text, list):
                    for point in reasoning_text:
                        lines.append(f"- {point}")
                else:
                    # Format as bullet points
                    for line in reasoning_text.split('\n'):
                        line = line.strip()
                        if line:
                            # Remove existing bullets and re-add consistently
                            line = line.lstrip('-').lstrip('•').lstrip('*').strip()
                            if line:
                                lines.append(f"- {line}")
            lines.append("")

        lines.append("---")
        lines.append("")

        # Current Price Section
        lines.append("## Current Price Information")
        lines.append("")
        lines.append(f"| Metric | Value |")
        lines.append(f"|--------|-------|")
        lines.append(f"| Current Price | {self._format_price(current_price.get('current_price'))} |")
        lines.append(f"| Day Change | {self._format_percent(current_price.get('day_change_percent'))} |")
        lines.append(f"| 52-Week High | {self._format_price(current_price.get('fifty_two_week_high'))} |")
        lines.append(f"| 52-Week Low | {self._format_price(current_price.get('fifty_two_week_low'))} |")
        lines.append(f"| Market Cap | {self._format_number(current_price.get('market_cap'))} |")
        lines.append("")

        # Price Chart
        if 'price_chart' in charts:
            lines.append("### Price History with Moving Averages")
            lines.append("")
            if save_charts:
                # Use relative path from report location
                chart_path = os.path.basename(charts['price_chart'])
                lines.append(f"![Price Chart]({chart_path})")
            else:
                # Embed base64
                lines.append(f"![Price Chart]({charts['price_chart']})")
            lines.append("")

        # Financial Metrics
        lines.append("## Financial Metrics")
        lines.append("")
        lines.append("| Metric | Value |")
        lines.append("|--------|-------|")
        lines.append(f"| P/E Ratio | {self._format_number(financials.get('pe_ratio'))} |")
        lines.append(f"| Forward P/E | {self._format_number(financials.get('forward_pe'))} |")
        lines.append(f"| EPS (TTM) | {self._format_price(financials.get('eps'))} |")
        lines.append(f"| Revenue Growth | {self._format_percent(financials.get('revenue_growth'))} |")
        lines.append(f"| Profit Margin | {self._format_percent(financials.get('profit_margin'))} |")
        lines.append(f"| Gross Margin | {self._format_percent(financials.get('gross_margin'))} |")
        lines.append(f"| Debt to Equity | {self._format_number(financials.get('debt_to_equity'))} |")
        lines.append(f"| ROE | {self._format_percent(financials.get('roe'))} |")
        lines.append(f"| Dividend Yield | {self._format_percent(financials.get('dividend_yield'))} |")
        lines.append(f"| Beta | {self._format_number(financials.get('beta'))} |")
        lines.append("")

        # Technical Indicators
        lines.append("## Technical Indicators")
        lines.append("")
        lines.append("| Indicator | Value |")
        lines.append("|-----------|-------|")
        lines.append(f"| 50-Day MA | {self._format_price(technicals.get('ma_50'))} |")
        lines.append(f"| 200-Day MA | {self._format_price(technicals.get('ma_200'))} |")
        lines.append(f"| RSI (14) | {self._format_number(technicals.get('rsi'))} |")
        lines.append(f"| MACD | {self._format_number(technicals.get('macd'))} |")
        lines.append(f"| Signal Line | {self._format_number(technicals.get('macd_signal'))} |")
        lines.append(f"| MACD Histogram | {self._format_number(technicals.get('macd_histogram'))} |")
        lines.append("")

        # Technical Charts
        if 'rsi_chart' in charts or 'macd_chart' in charts:
            lines.append("### Technical Charts")
            lines.append("")

            if 'rsi_chart' in charts:
                if save_charts:
                    chart_path = os.path.basename(charts['rsi_chart'])
                    lines.append(f"![RSI Chart]({chart_path})")
                else:
                    lines.append(f"![RSI Chart]({charts['rsi_chart']})")
                lines.append("")

            if 'macd_chart' in charts:
                if save_charts:
                    chart_path = os.path.basename(charts['macd_chart'])
                    lines.append(f"![MACD Chart]({chart_path})")
                else:
                    lines.append(f"![MACD Chart]({charts['macd_chart']})")
                lines.append("")

        # Volume Chart
        if 'volume_chart' in charts:
            lines.append("### Trading Volume")
            lines.append("")
            if save_charts:
                chart_path = os.path.basename(charts['volume_chart'])
                lines.append(f"![Volume Chart]({chart_path})")
            else:
                lines.append(f"![Volume Chart]({charts['volume_chart']})")
            lines.append("")

        # Analyst Ratings
        if analyst_ratings:
            lines.append("## Analyst Ratings")
            lines.append("")
            lines.append("| Metric | Value |")
            lines.append("|--------|-------|")
            lines.append(f"| Recommendation | {analyst_ratings.get('recommendation', 'N/A')} |")
            lines.append(f"| Number of Analysts | {analyst_ratings.get('num_analysts', 'N/A')} |")
            lines.append(f"| Mean Price Target | {self._format_price(analyst_ratings.get('target_mean'))} |")
            lines.append(f"| Low Price Target | {self._format_price(analyst_ratings.get('target_low'))} |")
            lines.append(f"| High Price Target | {self._format_price(analyst_ratings.get('target_high'))} |")
            lines.append("")

        # News Headlines
        if news:
            lines.append("## Recent News")
            lines.append("")
            for item in news[:10]:  # Limit to 10 news items
                title = item.get('title', 'No title')
                source = item.get('publisher', 'Unknown')
                link = item.get('link', '#')
                date = item.get('published', '')
                if date:
                    lines.append(f"- **[{title}]({link})** - {source} ({date})")
                else:
                    lines.append(f"- **[{title}]({link})** - {source}")
            lines.append("")

        # Insider Trading
        if insider:
            lines.append("## Recent Insider Activity")
            lines.append("")
            lines.append("| Insider | Transaction | Shares | Value | Date |")
            lines.append("|---------|-------------|--------|-------|------|")
            for trade in insider[:10]:  # Limit to 10 trades
                name = trade.get('insider', 'Unknown')
                transaction = trade.get('transaction', 'Unknown')
                shares = self._format_number(trade.get('shares'))
                value = self._format_price(trade.get('value'))
                date = trade.get('date', 'N/A')
                lines.append(f"| {name} | {transaction} | {shares} | {value} | {date} |")
            lines.append("")

        # Footer
        lines.append("---")
        lines.append("")
        lines.append("*This report was generated by Stock Analyzer using AI-powered analysis.*")
        lines.append("*Investment decisions should be made based on your own research and risk tolerance.*")
        lines.append("")
        lines.append(f"**LLM Provider:** {analysis.get('llm_provider', 'N/A')}")
        lines.append(f"**Model:** {analysis.get('llm_model', 'N/A')}")
        lines.append("")

        return "\n".join(lines)

    def get_report_path(self, ticker: str, date: Optional[str] = None) -> Optional[str]:
        """Get the path to a report file.

        Args:
            ticker: Stock ticker symbol
            date: Date string (YYYY-MM-DD), defaults to today

        Returns:
            Path to report if exists, None otherwise
        """
        ticker = ticker.upper()
        ticker_dir = os.path.join(self.reports_dir, ticker)

        if not os.path.exists(ticker_dir):
            return None

        if date:
            report_path = os.path.join(ticker_dir, f"{date}_analysis.md")
            return report_path if os.path.exists(report_path) else None

        # Get most recent report
        reports = [f for f in os.listdir(ticker_dir) if f.endswith('_analysis.md')]
        if not reports:
            return None

        reports.sort(reverse=True)
        return os.path.join(ticker_dir, reports[0])

    def get_report_content(self, report_path: str) -> Optional[str]:
        """Read and return report content.

        Args:
            report_path: Path to the report file

        Returns:
            Report content as string, None if not found
        """
        if not report_path or not os.path.exists(report_path):
            return None

        try:
            with open(report_path, 'r', encoding='utf-8') as f:
                return f.read()
        except Exception as e:
            logger.error(f"Failed to read report {report_path}: {e}")
            return None

    def list_reports(self, ticker: Optional[str] = None) -> list:
        """List all available reports.

        Args:
            ticker: Optional ticker to filter by

        Returns:
            List of report info dicts
        """
        reports = []

        if ticker:
            tickers = [ticker.upper()]
        else:
            if not os.path.exists(self.reports_dir):
                return []
            tickers = [d for d in os.listdir(self.reports_dir)
                      if os.path.isdir(os.path.join(self.reports_dir, d))]

        for t in tickers:
            ticker_dir = os.path.join(self.reports_dir, t)
            if not os.path.exists(ticker_dir):
                continue

            for filename in os.listdir(ticker_dir):
                if filename.endswith('_analysis.md'):
                    date_str = filename.replace('_analysis.md', '')
                    reports.append({
                        'ticker': t,
                        'date': date_str,
                        'filename': filename,
                        'path': os.path.join(ticker_dir, filename)
                    })

        # Sort by date descending
        reports.sort(key=lambda x: x['date'], reverse=True)
        return reports


# Global instance
report_generator = ReportGenerator()
