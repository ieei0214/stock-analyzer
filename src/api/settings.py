"""
Settings API endpoints.

Provides endpoints for managing application settings.
"""

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Optional
import os
from pathlib import Path
from dotenv import load_dotenv, set_key

from ..database.dao import SettingsDAO
from ..database.models import Settings, SettingsUpdate, APIResponse

router = APIRouter()

# Path to .env file
ENV_PATH = Path(__file__).parent.parent.parent / ".env"


class TestAPIRequest(BaseModel):
    """Request for testing API connection."""
    provider: str
    api_key: str


def mask_api_key(key: Optional[str]) -> str:
    """Mask an API key for display."""
    if not key or len(key) < 8:
        return ""
    return f"{key[:4]}...{key[-4:]}"


@router.get("", response_model=Settings)
async def get_settings():
    """Get current application settings."""
    # Load from environment
    load_dotenv(ENV_PATH)

    openai_key = os.getenv("OPENAI_API_KEY", "")
    gemini_key = os.getenv("GEMINI_API_KEY", "")

    # Determine if keys are set (not placeholder values)
    openai_key_set = bool(openai_key and openai_key != "your-openai-api-key-here")
    gemini_key_set = bool(gemini_key and gemini_key != "your-gemini-api-key-here")

    return Settings(
        llm_provider=os.getenv("LLM_PROVIDER", "openai"),
        openai_api_key_set=openai_key_set,
        gemini_api_key_set=gemini_key_set,
        openai_api_key_masked=mask_api_key(openai_key) if openai_key_set else None,
        gemini_api_key_masked=mask_api_key(gemini_key) if gemini_key_set else None,
        model_name=os.getenv("MODEL_NAME", "gpt-4"),
        default_analysis_style=os.getenv("DEFAULT_ANALYSIS_STYLE", "Conservative"),
        cache_duration_hours=int(os.getenv("CACHE_DURATION_HOURS", "1"))
    )


@router.put("", response_model=APIResponse)
async def update_settings(settings: SettingsUpdate):
    """Update application settings."""
    try:
        # Ensure .env exists
        if not ENV_PATH.exists():
            ENV_PATH.touch()

        # Update .env file
        if settings.llm_provider:
            set_key(str(ENV_PATH), "LLM_PROVIDER", settings.llm_provider)

        if settings.openai_api_key:
            set_key(str(ENV_PATH), "OPENAI_API_KEY", settings.openai_api_key)

        if settings.gemini_api_key:
            set_key(str(ENV_PATH), "GEMINI_API_KEY", settings.gemini_api_key)

        if settings.model_name:
            set_key(str(ENV_PATH), "MODEL_NAME", settings.model_name)

        if settings.default_analysis_style:
            set_key(str(ENV_PATH), "DEFAULT_ANALYSIS_STYLE", settings.default_analysis_style)

        if settings.cache_duration_hours is not None:
            set_key(str(ENV_PATH), "CACHE_DURATION_HOURS", str(settings.cache_duration_hours))

        # Reload environment
        load_dotenv(ENV_PATH, override=True)

        return APIResponse(success=True, message="Settings saved successfully")

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to save settings: {str(e)}")


@router.post("/test-api", response_model=APIResponse)
async def test_api_connection(request: TestAPIRequest):
    """Test LLM API connection."""
    provider = request.provider.lower()
    api_key = request.api_key

    if not api_key:
        raise HTTPException(status_code=400, detail="API key is required")

    if provider == "openai":
        try:
            import openai
            client = openai.OpenAI(api_key=api_key)
            # Make a simple API call to verify the key
            response = client.models.list()
            return APIResponse(
                success=True,
                message="OpenAI API connection successful",
                data={"models_available": len(response.data)}
            )
        except Exception as e:
            error_msg = str(e)
            if "invalid_api_key" in error_msg.lower() or "authentication" in error_msg.lower():
                return APIResponse(success=False, message="Invalid API key")
            return APIResponse(success=False, message=f"Connection failed: {error_msg}")

    elif provider == "gemini":
        try:
            import google.generativeai as genai
            genai.configure(api_key=api_key)
            # List models to verify the key
            models = list(genai.list_models())
            return APIResponse(
                success=True,
                message="Gemini API connection successful",
                data={"models_available": len(models)}
            )
        except Exception as e:
            error_msg = str(e)
            if "api_key" in error_msg.lower() or "authentication" in error_msg.lower():
                return APIResponse(success=False, message="Invalid API key")
            return APIResponse(success=False, message=f"Connection failed: {error_msg}")

    else:
        raise HTTPException(status_code=400, detail=f"Unknown provider: {provider}")


@router.post("/reset", response_model=APIResponse)
async def reset_settings():
    """Reset all settings to defaults."""
    try:
        # Default values
        defaults = {
            "LLM_PROVIDER": "openai",
            "MODEL_NAME": "gpt-4",
            "DEFAULT_ANALYSIS_STYLE": "Conservative",
            "CACHE_DURATION_HOURS": "1"
        }

        # Update .env file with defaults
        for key, value in defaults.items():
            set_key(str(ENV_PATH), key, value)

        # Reload environment
        load_dotenv(ENV_PATH, override=True)

        return APIResponse(success=True, message="Settings reset to defaults")

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to reset settings: {str(e)}")
