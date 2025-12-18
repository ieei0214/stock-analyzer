"""
Investment Analyst Agent.

This agent uses LLM (OpenAI or Gemini) to analyze stock data
and provide Buy/Sell/Hold recommendations with reasoning.
"""

import os
import json
import logging
from typing import Dict, Any, Optional, Tuple
from datetime import datetime

logger = logging.getLogger(__name__)


class AnalysisResult:
    """Structure for analysis results."""

    def __init__(
        self,
        ticker: str,
        recommendation: str,
        confidence: str,
        reasoning: str,
        price_at_analysis: float,
        analysis_style: str,
        llm_provider: str,
        llm_model: str,
    ):
        self.ticker = ticker
        self.recommendation = recommendation
        self.confidence = confidence
        self.reasoning = reasoning
        self.price_at_analysis = price_at_analysis
        self.analysis_style = analysis_style
        self.llm_provider = llm_provider
        self.llm_model = llm_model

    def to_dict(self) -> Dict[str, Any]:
        return {
            "ticker": self.ticker,
            "recommendation": self.recommendation,
            "confidence_level": self.confidence,
            "reasoning": self.reasoning,
            "price_at_analysis": self.price_at_analysis,
            "analysis_style": self.analysis_style,
            "llm_provider": self.llm_provider,
            "llm_model": self.llm_model,
        }


class InvestmentAnalystAgent:
    """
    Agent for performing LLM-powered investment analysis.

    Supports both OpenAI and Google Gemini as LLM providers.
    """

    # System prompt for the LLM
    SYSTEM_PROMPT = """You are a professional investment analyst providing stock recommendations.
You analyze stocks based on fundamental analysis, technical indicators, and market sentiment.

Your analysis style is: {style}

For Conservative analysis:
- Focus on long-term value and stability
- Emphasize strong fundamentals (low P/E, strong balance sheet)
- Be cautious about high volatility or debt
- Prefer established companies with consistent earnings

For Aggressive analysis:
- Focus on growth potential and momentum
- Willing to accept higher P/E for growth stocks
- Consider technical momentum indicators more heavily
- Accept higher risk for higher potential returns

You MUST respond with valid JSON in exactly this format:
{{
    "recommendation": "Buy" | "Sell" | "Hold",
    "confidence": "High" | "Medium" | "Low",
    "reasoning": "Your detailed analysis in bullet points, using - for each point"
}}

Be specific and cite actual numbers from the data provided. Keep reasoning concise but informative."""

    def __init__(
        self,
        provider: str = "openai",
        api_key: Optional[str] = None,
        model: Optional[str] = None
    ):
        """
        Initialize the analyst agent.

        Args:
            provider: LLM provider ("openai" or "gemini")
            api_key: API key for the provider
            model: Model name to use
        """
        self.provider = provider.lower()
        self.api_key = api_key or self._get_api_key()
        self.model = model or self._get_default_model()

        if not self.api_key:
            raise ValueError(f"No API key provided for {provider}")

    def _get_api_key(self) -> Optional[str]:
        """Get API key from environment."""
        if self.provider == "openai":
            return os.getenv("OPENAI_API_KEY")
        elif self.provider == "gemini":
            return os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
        return None

    def _get_default_model(self) -> str:
        """Get default model for the provider."""
        if self.provider == "openai":
            return "gpt-4"
        elif self.provider == "gemini":
            return "gemini-pro"
        return "gpt-4"

    def _format_data_for_prompt(self, stock_data: Dict[str, Any]) -> str:
        """Format stock data into a readable prompt."""
        company = stock_data.get("company", {})
        price = stock_data.get("price", {})
        financials = stock_data.get("financials", {})
        technicals = stock_data.get("technicals", {})
        analyst = stock_data.get("analyst", {})
        news = stock_data.get("news", [])

        # Format price info
        price_info = f"""
CURRENT PRICE DATA:
- Current Price: ${price.get('current_price', 'N/A')}
- Previous Close: ${price.get('previous_close', 'N/A')}
- Day Change: {price.get('change_percent', 'N/A'):.2f}% if price.get('change_percent') else 'N/A'
- 52-Week High: ${price.get('fifty_two_week_high', 'N/A')}
- 52-Week Low: ${price.get('fifty_two_week_low', 'N/A')}
- Market Cap: ${price.get('market_cap', 'N/A'):,} if price.get('market_cap') else 'N/A'
"""

        # Format financial metrics
        def fmt(val, is_pct=False, decimals=2):
            if val is None:
                return "N/A"
            if is_pct:
                return f"{val*100:.1f}%"
            return f"{val:.{decimals}f}"

        financial_info = f"""
FUNDAMENTAL METRICS:
Valuation:
- P/E Ratio (TTM): {fmt(financials.get('pe_ratio'))}
- Forward P/E: {fmt(financials.get('forward_pe'))}
- PEG Ratio: {fmt(financials.get('peg_ratio'))}
- Price to Book: {fmt(financials.get('price_to_book'))}
- Price to Sales: {fmt(financials.get('price_to_sales'))}

Earnings & Revenue:
- EPS (TTM): ${fmt(financials.get('eps'))}
- Forward EPS: ${fmt(financials.get('forward_eps'))}
- Earnings Growth: {fmt(financials.get('earnings_growth'), True)}
- Revenue Growth: {fmt(financials.get('revenue_growth'), True)}

Profitability:
- Gross Margin: {fmt(financials.get('gross_margin'), True)}
- Operating Margin: {fmt(financials.get('operating_margin'), True)}
- Profit Margin: {fmt(financials.get('profit_margin'), True)}
- ROE: {fmt(financials.get('return_on_equity'), True)}
- ROA: {fmt(financials.get('return_on_assets'), True)}

Financial Health:
- Debt to Equity: {fmt(financials.get('debt_to_equity'))}
- Current Ratio: {fmt(financials.get('current_ratio'))}
- Quick Ratio: {fmt(financials.get('quick_ratio'))}

Dividends:
- Dividend Yield: {fmt(financials.get('dividend_yield'), True)}
- Payout Ratio: {fmt(financials.get('payout_ratio'), True)}

Risk:
- Beta: {fmt(financials.get('beta'))}
"""

        # Format technical indicators
        technical_info = f"""
TECHNICAL INDICATORS:
Moving Averages:
- 50-Day MA: ${fmt(technicals.get('ma_50'))}
- 200-Day MA: ${fmt(technicals.get('ma_200'))}
- MA Signal: {technicals.get('ma_cross_signal', 'N/A')}
- Price vs 50 MA: {'Above' if technicals.get('current_price') and technicals.get('ma_50') and technicals.get('current_price') > technicals.get('ma_50') else 'Below'}

Momentum:
- RSI (14): {fmt(technicals.get('rsi_14'))}
- RSI Signal: {technicals.get('rsi_signal', 'N/A')}
- MACD: {fmt(technicals.get('macd'))}
- MACD Signal Line: {fmt(technicals.get('macd_signal'))}
- MACD Histogram: {fmt(technicals.get('macd_histogram'))}

Bollinger Bands:
- Upper: ${fmt(technicals.get('bollinger_upper'))}
- Middle: ${fmt(technicals.get('bollinger_middle'))}
- Lower: ${fmt(technicals.get('bollinger_lower'))}
"""

        # Format analyst ratings
        analyst_info = f"""
ANALYST RATINGS:
- Consensus: {analyst.get('recommendation_key', 'N/A')}
- Mean Rating: {fmt(analyst.get('recommendation_mean'))} (1=Strong Buy, 5=Strong Sell)
- Number of Analysts: {analyst.get('number_of_analysts', 'N/A')}
- Target Price (Mean): ${fmt(analyst.get('target_mean'))}
- Target Price (High): ${fmt(analyst.get('target_high'))}
- Target Price (Low): ${fmt(analyst.get('target_low'))}
"""

        # Format news headlines (just titles)
        news_headlines = "\n".join([f"- {n.get('title', '')}" for n in news[:5]]) if news else "No recent news"
        news_info = f"""
RECENT NEWS:
{news_headlines}
"""

        return f"""
STOCK: {stock_data.get('ticker')} - {company.get('company_name', 'Unknown Company')}
SECTOR: {company.get('sector', 'Unknown')} | INDUSTRY: {company.get('industry', 'Unknown')}
DATA AS OF: {stock_data.get('collected_at', datetime.now().isoformat())}

{price_info}
{financial_info}
{technical_info}
{analyst_info}
{news_info}
"""

    async def analyze_with_openai(
        self,
        stock_data: Dict[str, Any],
        style: str = "Conservative"
    ) -> AnalysisResult:
        """Analyze stock using OpenAI API."""
        try:
            from openai import OpenAI

            client = OpenAI(api_key=self.api_key)

            data_prompt = self._format_data_for_prompt(stock_data)
            system_prompt = self.SYSTEM_PROMPT.format(style=style)

            response = client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": f"Analyze this stock and provide your recommendation:\n\n{data_prompt}"}
                ],
                temperature=0.3,
                max_tokens=1000,
            )

            result_text = response.choices[0].message.content
            return self._parse_llm_response(result_text, stock_data, style)

        except Exception as e:
            logger.error(f"OpenAI API error: {e}")
            raise

    async def analyze_with_gemini(
        self,
        stock_data: Dict[str, Any],
        style: str = "Conservative"
    ) -> AnalysisResult:
        """Analyze stock using Google Gemini API."""
        try:
            import google.generativeai as genai

            genai.configure(api_key=self.api_key)
            model = genai.GenerativeModel(self.model)

            data_prompt = self._format_data_for_prompt(stock_data)
            system_prompt = self.SYSTEM_PROMPT.format(style=style)

            full_prompt = f"{system_prompt}\n\nAnalyze this stock and provide your recommendation:\n\n{data_prompt}"

            response = model.generate_content(
                full_prompt,
                generation_config=genai.types.GenerationConfig(
                    temperature=0.3,
                    max_output_tokens=1000,
                )
            )

            result_text = response.text
            return self._parse_llm_response(result_text, stock_data, style)

        except Exception as e:
            logger.error(f"Gemini API error: {e}")
            raise

    def _parse_llm_response(
        self,
        response_text: str,
        stock_data: Dict[str, Any],
        style: str
    ) -> AnalysisResult:
        """Parse the LLM response into an AnalysisResult."""
        try:
            # Try to extract JSON from the response
            # Sometimes LLMs wrap JSON in markdown code blocks
            text = response_text.strip()
            if "```json" in text:
                text = text.split("```json")[1].split("```")[0].strip()
            elif "```" in text:
                text = text.split("```")[1].split("```")[0].strip()

            result = json.loads(text)

            recommendation = result.get("recommendation", "Hold")
            confidence = result.get("confidence", "Medium")
            reasoning = result.get("reasoning", "No detailed reasoning provided.")

            # Validate recommendation
            if recommendation not in ("Buy", "Sell", "Hold"):
                recommendation = "Hold"

            if confidence not in ("High", "Medium", "Low"):
                confidence = "Medium"

            price = stock_data.get("price", {}).get("current_price", 0)

            return AnalysisResult(
                ticker=stock_data.get("ticker", "UNKNOWN"),
                recommendation=recommendation,
                confidence=confidence,
                reasoning=reasoning,
                price_at_analysis=price,
                analysis_style=style,
                llm_provider=self.provider,
                llm_model=self.model,
            )

        except json.JSONDecodeError as e:
            logger.warning(f"Failed to parse LLM response as JSON: {e}")
            # Fallback: try to extract information from plain text
            return self._parse_plain_text_response(response_text, stock_data, style)

    def _parse_plain_text_response(
        self,
        response_text: str,
        stock_data: Dict[str, Any],
        style: str
    ) -> AnalysisResult:
        """Fallback parser for non-JSON responses."""
        text_lower = response_text.lower()

        # Determine recommendation from text
        if "strong buy" in text_lower or "strongly recommend buying" in text_lower:
            recommendation = "Buy"
            confidence = "High"
        elif "buy" in text_lower and "don't buy" not in text_lower:
            recommendation = "Buy"
            confidence = "Medium"
        elif "strong sell" in text_lower or "strongly recommend selling" in text_lower:
            recommendation = "Sell"
            confidence = "High"
        elif "sell" in text_lower and "don't sell" not in text_lower:
            recommendation = "Sell"
            confidence = "Medium"
        else:
            recommendation = "Hold"
            confidence = "Medium"

        price = stock_data.get("price", {}).get("current_price", 0)

        return AnalysisResult(
            ticker=stock_data.get("ticker", "UNKNOWN"),
            recommendation=recommendation,
            confidence=confidence,
            reasoning=response_text,
            price_at_analysis=price,
            analysis_style=style,
            llm_provider=self.provider,
            llm_model=self.model,
        )

    async def analyze(
        self,
        stock_data: Dict[str, Any],
        style: str = "Conservative"
    ) -> AnalysisResult:
        """
        Perform investment analysis on the stock data.

        Args:
            stock_data: Comprehensive stock data from DataCollectorAgent
            style: Analysis style ("Conservative" or "Aggressive")

        Returns:
            AnalysisResult with recommendation and reasoning
        """
        if self.provider == "openai":
            return await self.analyze_with_openai(stock_data, style)
        elif self.provider == "gemini":
            return await self.analyze_with_gemini(stock_data, style)
        else:
            raise ValueError(f"Unsupported LLM provider: {self.provider}")


async def analyze_stock(
    stock_data: Dict[str, Any],
    provider: str = "openai",
    api_key: Optional[str] = None,
    model: Optional[str] = None,
    style: str = "Conservative"
) -> AnalysisResult:
    """
    Convenience function to analyze a stock.

    Args:
        stock_data: Stock data from DataCollectorAgent
        provider: LLM provider
        api_key: API key
        model: Model name
        style: Analysis style

    Returns:
        AnalysisResult
    """
    analyst = InvestmentAnalystAgent(provider=provider, api_key=api_key, model=model)
    return await analyst.analyze(stock_data, style)
