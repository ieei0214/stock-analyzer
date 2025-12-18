# Stock Analyzer - Chart Generator
"""
Generate stock analysis charts using matplotlib.

Charts include:
- Price history with moving averages
- Volume chart
- RSI indicator
- MACD with signal line
"""

import os
import io
import base64
from datetime import datetime
from typing import Dict, Any, Optional, List
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np


class ChartGenerator:
    """Generate stock analysis charts."""

    def __init__(self, reports_dir: str = "reports"):
        """Initialize chart generator.

        Args:
            reports_dir: Directory to save chart images
        """
        self.reports_dir = reports_dir

        # Style settings
        plt.style.use('seaborn-v0_8-whitegrid')
        self.colors = {
            'price': '#3B82F6',      # Primary blue
            'ma_50': '#F59E0B',       # Warning yellow
            'ma_200': '#EF4444',      # Danger red
            'volume_up': '#10B981',   # Success green
            'volume_down': '#EF4444', # Danger red
            'rsi': '#8B5CF6',         # Purple
            'macd': '#3B82F6',        # Blue
            'signal': '#F59E0B',      # Yellow
            'histogram_pos': '#10B981',  # Green
            'histogram_neg': '#EF4444',  # Red
        }

    def _ensure_dir(self, ticker: str) -> str:
        """Ensure report directory exists for ticker."""
        ticker_dir = os.path.join(self.reports_dir, ticker.upper())
        os.makedirs(ticker_dir, exist_ok=True)
        return ticker_dir

    def _save_or_encode(self, fig: plt.Figure, ticker: str, chart_name: str,
                        save_to_file: bool = False) -> str:
        """Save chart to file or return base64 encoded image.

        Args:
            fig: Matplotlib figure
            ticker: Stock ticker
            chart_name: Name of the chart
            save_to_file: Whether to save to file

        Returns:
            File path or base64 encoded image
        """
        if save_to_file:
            ticker_dir = self._ensure_dir(ticker)
            date_str = datetime.now().strftime("%Y-%m-%d")
            filepath = os.path.join(ticker_dir, f"{date_str}_{chart_name}.png")
            fig.savefig(filepath, dpi=150, bbox_inches='tight',
                       facecolor='white', edgecolor='none')
            plt.close(fig)
            return filepath
        else:
            # Return base64 encoded image
            buf = io.BytesIO()
            fig.savefig(buf, format='png', dpi=150, bbox_inches='tight',
                       facecolor='white', edgecolor='none')
            buf.seek(0)
            img_base64 = base64.b64encode(buf.read()).decode('utf-8')
            plt.close(fig)
            return f"data:image/png;base64,{img_base64}"

    def generate_price_chart(self, price_data: Dict[str, Any],
                            technicals: Optional[Dict[str, Any]] = None,
                            ticker: str = "STOCK",
                            save_to_file: bool = False) -> str:
        """Generate price history chart with moving averages.

        Args:
            price_data: Price history data with dates and prices
            technicals: Technical indicators including MAs
            ticker: Stock ticker symbol
            save_to_file: Whether to save to file

        Returns:
            File path or base64 encoded image
        """
        fig, ax = plt.subplots(figsize=(12, 6))

        # Extract price data
        dates = price_data.get('dates', [])
        closes = price_data.get('close', [])

        if not dates or not closes:
            # Create placeholder chart
            ax.text(0.5, 0.5, 'No price data available',
                   ha='center', va='center', fontsize=14)
            ax.set_title(f'{ticker} - Price History')
            return self._save_or_encode(fig, ticker, 'price_chart', save_to_file)

        # Convert dates if needed
        if isinstance(dates[0], str):
            dates = [datetime.fromisoformat(d.replace('Z', '+00:00'))
                    if 'T' in d else datetime.strptime(d, '%Y-%m-%d')
                    for d in dates]

        # Plot close prices
        ax.plot(dates, closes, color=self.colors['price'],
               linewidth=1.5, label='Close Price')

        # Plot moving averages if available
        if technicals:
            ma_50 = technicals.get('ma_50_history', [])
            ma_200 = technicals.get('ma_200_history', [])

            # Align MAs with dates (they're calculated from recent data)
            if ma_50 and len(ma_50) <= len(dates):
                ma_dates = dates[-len(ma_50):]
                # Filter out None values
                valid_ma_50 = [(d, v) for d, v in zip(ma_dates, ma_50) if v is not None]
                if valid_ma_50:
                    ma_dates_50, ma_values_50 = zip(*valid_ma_50)
                    ax.plot(ma_dates_50, ma_values_50, color=self.colors['ma_50'],
                           linewidth=1, linestyle='--', label='50-Day MA', alpha=0.8)

            if ma_200 and len(ma_200) <= len(dates):
                ma_dates = dates[-len(ma_200):]
                valid_ma_200 = [(d, v) for d, v in zip(ma_dates, ma_200) if v is not None]
                if valid_ma_200:
                    ma_dates_200, ma_values_200 = zip(*valid_ma_200)
                    ax.plot(ma_dates_200, ma_values_200, color=self.colors['ma_200'],
                           linewidth=1, linestyle='--', label='200-Day MA', alpha=0.8)

        # Formatting
        ax.set_title(f'{ticker} - Price History', fontsize=14, fontweight='bold')
        ax.set_xlabel('Date', fontsize=10)
        ax.set_ylabel('Price ($)', fontsize=10)
        ax.legend(loc='upper left', framealpha=0.9)
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
        ax.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
        plt.xticks(rotation=45)

        # Add grid
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        return self._save_or_encode(fig, ticker, 'price_chart', save_to_file)

    def generate_volume_chart(self, price_data: Dict[str, Any],
                             ticker: str = "STOCK",
                             save_to_file: bool = False) -> str:
        """Generate volume chart.

        Args:
            price_data: Price history data with dates and volume
            ticker: Stock ticker symbol
            save_to_file: Whether to save to file

        Returns:
            File path or base64 encoded image
        """
        fig, ax = plt.subplots(figsize=(12, 4))

        dates = price_data.get('dates', [])
        volumes = price_data.get('volume', [])
        opens = price_data.get('open', [])
        closes = price_data.get('close', [])

        if not dates or not volumes:
            ax.text(0.5, 0.5, 'No volume data available',
                   ha='center', va='center', fontsize=14)
            ax.set_title(f'{ticker} - Trading Volume')
            return self._save_or_encode(fig, ticker, 'volume_chart', save_to_file)

        # Convert dates if needed
        if isinstance(dates[0], str):
            dates = [datetime.fromisoformat(d.replace('Z', '+00:00'))
                    if 'T' in d else datetime.strptime(d, '%Y-%m-%d')
                    for d in dates]

        # Color bars based on price direction
        colors = []
        for i in range(len(volumes)):
            if i < len(opens) and i < len(closes):
                if closes[i] >= opens[i]:
                    colors.append(self.colors['volume_up'])
                else:
                    colors.append(self.colors['volume_down'])
            else:
                colors.append(self.colors['volume_up'])

        ax.bar(dates, volumes, color=colors, alpha=0.7, width=0.8)

        # Formatting
        ax.set_title(f'{ticker} - Trading Volume', fontsize=14, fontweight='bold')
        ax.set_xlabel('Date', fontsize=10)
        ax.set_ylabel('Volume', fontsize=10)
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
        ax.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
        plt.xticks(rotation=45)

        # Format y-axis with millions
        ax.yaxis.set_major_formatter(plt.FuncFormatter(
            lambda x, p: f'{x/1e6:.1f}M' if x >= 1e6 else f'{x/1e3:.0f}K'))

        ax.grid(True, alpha=0.3, axis='y')
        plt.tight_layout()

        return self._save_or_encode(fig, ticker, 'volume_chart', save_to_file)

    def generate_rsi_chart(self, technicals: Dict[str, Any],
                          dates: List[Any] = None,
                          ticker: str = "STOCK",
                          save_to_file: bool = False) -> str:
        """Generate RSI indicator chart.

        Args:
            technicals: Technical indicators including RSI history
            dates: List of dates for x-axis
            ticker: Stock ticker symbol
            save_to_file: Whether to save to file

        Returns:
            File path or base64 encoded image
        """
        fig, ax = plt.subplots(figsize=(12, 4))

        rsi_history = technicals.get('rsi_history', [])

        if not rsi_history:
            ax.text(0.5, 0.5, 'No RSI data available',
                   ha='center', va='center', fontsize=14)
            ax.set_title(f'{ticker} - RSI (14)')
            return self._save_or_encode(fig, ticker, 'rsi_chart', save_to_file)

        # Create x-axis (use dates if provided, otherwise sequential)
        if dates and len(dates) >= len(rsi_history):
            x_data = dates[-len(rsi_history):]
            if isinstance(x_data[0], str):
                x_data = [datetime.fromisoformat(d.replace('Z', '+00:00'))
                         if 'T' in d else datetime.strptime(d, '%Y-%m-%d')
                         for d in x_data]
        else:
            x_data = list(range(len(rsi_history)))

        # Plot RSI line
        ax.plot(x_data, rsi_history, color=self.colors['rsi'], linewidth=1.5)

        # Add overbought/oversold levels
        ax.axhline(y=70, color='red', linestyle='--', alpha=0.5, label='Overbought (70)')
        ax.axhline(y=30, color='green', linestyle='--', alpha=0.5, label='Oversold (30)')
        ax.axhline(y=50, color='gray', linestyle=':', alpha=0.3)

        # Fill overbought/oversold regions
        ax.fill_between(x_data, 70, [max(70, r) for r in rsi_history],
                       color='red', alpha=0.1)
        ax.fill_between(x_data, 30, [min(30, r) for r in rsi_history],
                       color='green', alpha=0.1)

        # Formatting
        ax.set_title(f'{ticker} - RSI (14-Day)', fontsize=14, fontweight='bold')
        ax.set_xlabel('Date', fontsize=10)
        ax.set_ylabel('RSI', fontsize=10)
        ax.set_ylim(0, 100)
        ax.legend(loc='upper right', framealpha=0.9, fontsize=8)

        if isinstance(x_data[0], datetime):
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
            ax.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
            plt.xticks(rotation=45)

        ax.grid(True, alpha=0.3)
        plt.tight_layout()

        return self._save_or_encode(fig, ticker, 'rsi_chart', save_to_file)

    def generate_macd_chart(self, technicals: Dict[str, Any],
                           dates: List[Any] = None,
                           ticker: str = "STOCK",
                           save_to_file: bool = False) -> str:
        """Generate MACD chart with signal line and histogram.

        Args:
            technicals: Technical indicators including MACD data
            dates: List of dates for x-axis
            ticker: Stock ticker symbol
            save_to_file: Whether to save to file

        Returns:
            File path or base64 encoded image
        """
        fig, ax = plt.subplots(figsize=(12, 4))

        macd_history = technicals.get('macd_history', [])
        signal_history = technicals.get('macd_signal_history', [])

        if not macd_history:
            ax.text(0.5, 0.5, 'No MACD data available',
                   ha='center', va='center', fontsize=14)
            ax.set_title(f'{ticker} - MACD')
            return self._save_or_encode(fig, ticker, 'macd_chart', save_to_file)

        # Create x-axis
        if dates and len(dates) >= len(macd_history):
            x_data = dates[-len(macd_history):]
            if isinstance(x_data[0], str):
                x_data = [datetime.fromisoformat(d.replace('Z', '+00:00'))
                         if 'T' in d else datetime.strptime(d, '%Y-%m-%d')
                         for d in x_data]
        else:
            x_data = list(range(len(macd_history)))

        # Calculate histogram
        histogram = []
        for i in range(len(macd_history)):
            if i < len(signal_history) and signal_history[i] is not None:
                histogram.append(macd_history[i] - signal_history[i])
            else:
                histogram.append(0)

        # Plot histogram bars
        colors = [self.colors['histogram_pos'] if h >= 0 else self.colors['histogram_neg']
                 for h in histogram]
        ax.bar(x_data, histogram, color=colors, alpha=0.5, width=0.8, label='Histogram')

        # Plot MACD and Signal lines
        ax.plot(x_data, macd_history, color=self.colors['macd'],
               linewidth=1.5, label='MACD')
        if signal_history:
            valid_signal = [(x, s) for x, s in zip(x_data, signal_history) if s is not None]
            if valid_signal:
                sig_x, sig_y = zip(*valid_signal)
                ax.plot(sig_x, sig_y, color=self.colors['signal'],
                       linewidth=1.5, label='Signal')

        # Zero line
        ax.axhline(y=0, color='gray', linestyle='-', alpha=0.5)

        # Formatting
        ax.set_title(f'{ticker} - MACD', fontsize=14, fontweight='bold')
        ax.set_xlabel('Date', fontsize=10)
        ax.set_ylabel('MACD', fontsize=10)
        ax.legend(loc='upper right', framealpha=0.9, fontsize=8)

        if isinstance(x_data[0], datetime):
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
            ax.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
            plt.xticks(rotation=45)

        ax.grid(True, alpha=0.3)
        plt.tight_layout()

        return self._save_or_encode(fig, ticker, 'macd_chart', save_to_file)

    def generate_all_charts(self, ticker: str, price_data: Dict[str, Any],
                           technicals: Dict[str, Any],
                           save_to_file: bool = False) -> Dict[str, str]:
        """Generate all charts for a stock analysis.

        Args:
            ticker: Stock ticker symbol
            price_data: Price history data
            technicals: Technical indicators
            save_to_file: Whether to save to file

        Returns:
            Dictionary of chart names to file paths or base64 images
        """
        dates = price_data.get('dates', [])

        return {
            'price_chart': self.generate_price_chart(
                price_data, technicals, ticker, save_to_file),
            'volume_chart': self.generate_volume_chart(
                price_data, ticker, save_to_file),
            'rsi_chart': self.generate_rsi_chart(
                technicals, dates, ticker, save_to_file),
            'macd_chart': self.generate_macd_chart(
                technicals, dates, ticker, save_to_file),
        }


# Global instance
chart_generator = ChartGenerator()
