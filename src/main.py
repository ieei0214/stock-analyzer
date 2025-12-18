"""
Stock Analyzer - Main FastAPI Application

Multi-agent stock analysis system with web dashboard.
"""

from fastapi import FastAPI, Request, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
from pathlib import Path
import logging

from .database.connection import init_database, check_database_exists
from .api import watchlist, analysis, stock, performance, settings, tasks

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Get project root directory
PROJECT_ROOT = Path(__file__).parent.parent


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager."""
    # Startup
    logger.info("Starting Stock Analyzer...")

    # Initialize database
    await init_database()
    logger.info("Database initialized")

    yield

    # Shutdown
    logger.info("Shutting down Stock Analyzer...")


# Create FastAPI application
app = FastAPI(
    title="Stock Analyzer",
    description="Multi-agent stock analysis system with web dashboard",
    version="1.0.0",
    lifespan=lifespan
)

# Add CORS middleware for local development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:8000", "http://127.0.0.1:8000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static files
static_path = PROJECT_ROOT / "static"
if static_path.exists():
    app.mount("/static", StaticFiles(directory=str(static_path)), name="static")

# Set up templates
templates_path = PROJECT_ROOT / "templates"
templates = Jinja2Templates(directory=str(templates_path))


# ===========================================
# Include API Routers
# ===========================================

app.include_router(watchlist.router, prefix="/api/watchlist", tags=["Watchlist"])
app.include_router(analysis.router, prefix="/api", tags=["Analysis"])
app.include_router(stock.router, prefix="/api/stock", tags=["Stock Data"])
app.include_router(performance.router, prefix="/api/performance", tags=["Performance"])
app.include_router(settings.router, prefix="/api/settings", tags=["Settings"])
app.include_router(tasks.router, prefix="/api/tasks", tags=["Background Tasks"])


# ===========================================
# HTML Page Routes
# ===========================================

@app.get("/", response_class=HTMLResponse)
async def dashboard(request: Request):
    """Dashboard page - Watchlist overview."""
    return templates.TemplateResponse(
        "dashboard.html",
        {"request": request, "page": "dashboard"}
    )


@app.get("/analyze", response_class=HTMLResponse)
async def analyze_page(request: Request):
    """Analyze Stock page."""
    return templates.TemplateResponse(
        "analyze.html",
        {"request": request, "page": "analyze"}
    )


@app.get("/history", response_class=HTMLResponse)
async def history_page(request: Request):
    """History and Performance page."""
    return templates.TemplateResponse(
        "history.html",
        {"request": request, "page": "history"}
    )


@app.get("/settings", response_class=HTMLResponse)
async def settings_page(request: Request):
    """Settings page."""
    return templates.TemplateResponse(
        "settings.html",
        {"request": request, "page": "settings"}
    )


@app.get("/analysis/{ticker}", response_class=HTMLResponse)
async def analysis_detail_page(request: Request, ticker: str):
    """Analysis detail page for a specific stock."""
    return templates.TemplateResponse(
        "analysis_detail.html",
        {"request": request, "page": "analysis", "ticker": ticker.upper()}
    )


# ===========================================
# Error Handlers
# ===========================================

@app.exception_handler(404)
async def not_found_handler(request: Request, exc: HTTPException):
    """Handle 404 errors."""
    if request.url.path.startswith("/api/"):
        return JSONResponse(
            status_code=404,
            content={"success": False, "message": "Resource not found"}
        )
    return templates.TemplateResponse(
        "404.html",
        {"request": request, "page": "error"},
        status_code=404
    )


@app.exception_handler(500)
async def server_error_handler(request: Request, exc: Exception):
    """Handle 500 errors."""
    logger.error(f"Server error: {exc}")
    if request.url.path.startswith("/api/"):
        return JSONResponse(
            status_code=500,
            content={"success": False, "message": "Internal server error"}
        )
    return templates.TemplateResponse(
        "500.html",
        {"request": request, "page": "error"},
        status_code=500
    )


# ===========================================
# Health Check
# ===========================================

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    db_exists = await check_database_exists()
    return {
        "status": "healthy",
        "database": "connected" if db_exists else "not initialized"
    }
