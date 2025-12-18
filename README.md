# Stock Analyzer

A Python-based multi-agent stock analysis system with a web dashboard interface. The system uses two specialized agents: a Data Collector that fetches comprehensive stock data and an LLM-powered Investment Analyst that evaluates the data to provide Buy/Sell/Hold recommendations with detailed reasoning.

## Features

- **Multi-Agent Architecture**: Two specialized agents working together
  - **Data Collector Agent**: Fetches price history, financials, news, insider trading, and analyst ratings
  - **Investment Analyst Agent**: LLM-powered analysis with Buy/Sell/Hold recommendations

- **Web Dashboard**
  - Watchlist management (add/remove stocks)
  - One-click stock analysis
  - Batch "Analyze All" for entire watchlist
  - Interactive charts (price history, RSI, MACD, volume)
  - Detailed reports with embedded visualizations

- **Historical Performance Tracking**
  - Track past recommendations
  - Calculate gain/loss performance
  - Success rate statistics

- **Flexible Configuration**
  - Support for OpenAI and Google Gemini LLMs
  - Conservative and Aggressive analysis styles
  - Configurable cache duration

## Technology Stack

- **Backend**: Python 3.10+, FastAPI
- **Frontend**: HTML/CSS/JavaScript with Tailwind CSS
- **Database**: SQLite
- **Stock Data**: yfinance
- **LLM**: OpenAI API / Google Gemini API
- **Charts**: Chart.js / Plotly.js, matplotlib

## Prerequisites

- Python 3.10 or higher
- pip package manager
- API key for OpenAI or Google Gemini
- Internet connection for fetching stock data

## Quick Start

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd stock-analyzer
   ```

2. **Run the setup script**
   ```bash
   ./init.sh
   ```

3. **Configure API keys**

   Edit the `.env` file and add your API keys:
   ```
   LLM_PROVIDER=openai
   OPENAI_API_KEY=your-openai-api-key-here
   # or
   LLM_PROVIDER=gemini
   GEMINI_API_KEY=your-gemini-api-key-here
   ```

4. **Start the development server**
   ```bash
   source venv/bin/activate
   uvicorn src.main:app --reload --host 127.0.0.1 --port 8000
   ```

5. **Open the application**

   Navigate to http://localhost:8000 in your browser

## Project Structure

```
stock-analyzer/
├── src/
│   ├── agents/           # Agent implementations
│   │   ├── data_collector.py
│   │   └── analyst.py
│   ├── api/              # FastAPI routes
│   │   ├── watchlist.py
│   │   ├── analysis.py
│   │   ├── stock.py
│   │   ├── performance.py
│   │   └── settings.py
│   ├── database/         # Database models and DAOs
│   │   ├── models.py
│   │   ├── connection.py
│   │   └── dao.py
│   ├── charts/           # Chart generation
│   │   └── generator.py
│   ├── reports/          # Report generation
│   │   └── generator.py
│   └── main.py           # FastAPI application entry
├── static/               # Static assets
│   ├── css/
│   └── js/
├── templates/            # HTML templates
├── data/                 # SQLite database
├── reports/              # Generated analysis reports
├── tests/                # Test suite
├── .env                  # Environment configuration
├── .env.example          # Environment template
├── requirements.txt      # Python dependencies
├── init.sh              # Setup script
└── README.md
```

## API Endpoints

### Watchlist
- `GET /api/watchlist` - Get all watchlist stocks
- `POST /api/watchlist` - Add stock to watchlist
- `DELETE /api/watchlist/{ticker}` - Remove stock from watchlist
- `POST /api/watchlist/analyze-all` - Trigger batch analysis

### Analysis
- `POST /api/analyze/{ticker}` - Analyze a single stock
- `GET /api/analysis/{ticker}` - Get latest analysis
- `GET /api/analysis/{ticker}/history` - Get analysis history

### Stock Data
- `GET /api/stock/{ticker}` - Get stock overview
- `GET /api/stock/{ticker}/data` - Get cached stock data
- `POST /api/stock/{ticker}/refresh` - Force data refresh

### Performance
- `POST /api/performance/check/{analysis_id}` - Check performance
- `GET /api/performance/stats` - Get performance statistics

### Settings
- `GET /api/settings` - Get current settings
- `PUT /api/settings` - Update settings
- `POST /api/settings/test-api` - Test LLM API connection

## Configuration

### Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `LLM_PROVIDER` | LLM provider (openai/gemini) | openai |
| `OPENAI_API_KEY` | OpenAI API key | - |
| `GEMINI_API_KEY` | Google Gemini API key | - |
| `MODEL_NAME` | Model to use | gpt-4 |
| `HOST` | Server host | 127.0.0.1 |
| `PORT` | Server port | 8000 |
| `CACHE_DURATION_HOURS` | Cache duration | 1 |
| `DEFAULT_ANALYSIS_STYLE` | Analysis style | Conservative |

## Usage

### Adding Stocks to Watchlist

1. Navigate to the Dashboard
2. Enter a ticker symbol (e.g., AAPL, MSFT, GOOGL)
3. Click "Add" to add to your watchlist

### Analyzing a Stock

1. Go to the "Analyze" page
2. Enter the ticker symbol
3. Select analysis style (Conservative/Aggressive)
4. Click "Analyze"
5. View the recommendation and detailed reasoning

### Batch Analysis

1. Add multiple stocks to your watchlist
2. Click "Analyze All" on the Dashboard
3. Monitor progress as each stock is analyzed

### Tracking Performance

1. Go to the "History" page
2. View past recommendations
3. Click "Check Performance" to see gain/loss

## Development

### Running Tests

```bash
source venv/bin/activate
pytest tests/ -v
```

### Code Style

The project follows PEP 8 style guidelines.

## License

MIT License

## Acknowledgments

- Stock data provided by [yfinance](https://github.com/ranaroussi/yfinance)
- LLM capabilities powered by OpenAI and Google Gemini
