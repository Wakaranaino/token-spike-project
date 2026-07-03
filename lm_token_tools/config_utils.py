"""
config_utils.py

Backend setup for LM token / CoT-token experiments.

Supported backends:
1. local
   - For LM Studio
   - No API key needed
   - Model can be auto-detected

2. api
   - For OpenAI-compatible cloud APIs
   - Examples: Groq, OpenAI-compatible lab server, other hosted model APIs
   - User provides API base URL, model name, and API key

This file only prepares backend settings.
It does not run the experiment.
"""

from dataclasses import dataclass
import requests


# =======================
# DEFAULT LOCAL SETTINGS
# =======================

@dataclass
class BackendConfig:
    backend_name: str
    base_url: str
    chat_url: str
    model: str
    headers: dict
    timeout: int


# =======================
# URL HELPERS
# =======================

def build_chat_url(base_url: str) -> str:
    """Build endpoint for sending chat prompts."""
    return f"{base_url.rstrip('/')}/v1/chat/completions"


def build_models_url(base_url: str) -> str:
    """Build endpoint for listing available models."""
    return f"{base_url.rstrip('/')}/v1/models"


# =======================
# MODEL DETECTION
# =======================

def get_first_available_model(base_url: str, headers: dict | None = None) -> str | None:
    """
    Try to detect the first available model from /v1/models.

    Mainly useful for LM Studio.
    For cloud APIs, users usually choose the model manually.
    """

    try:
        response = requests.get(
            build_models_url(base_url),
            headers=headers or {},
            timeout=10
        )
        response.raise_for_status()

        models = response.json().get("data", [])
        if not models:
            return None

        return models[0].get("id")

    except Exception:
        return None


# =======================
# MAIN CONFIG FUNCTION
# =======================

def get_backend_config(
    backend: str = "local",

    # local LM Studio settings
    local_base_url: str | None = None,
    local_model: str | None = None,
    auto_detect_local_model: bool = True,

    # generic API settings
    api_base_url: str | None = None,
    api_model: str | None = None,
    api_key: str | None = None,
    auth_type: str = "bearer",  # "bearer" for Groq/OpenAI-style, "api-key" for Azure
    ask_for_api_key: bool = True,

    # request setting
    timeout: int = 300
) -> BackendConfig:
    """
    Create backend config for model calls.

    backend:
        "local" = LM Studio
        "api"   = OpenAI-compatible cloud/server API

    For local:
        get_backend_config(
            backend="local",
            local_base_url="http://localhost:1234"
        )

    For generic API:
        get_backend_config(
            backend="api",
            api_base_url="https://api.groq.com/openai",
            api_model="llama-3.3-70b-versatile",
            api_key="paste_key_here"
        )

    If api_key is not provided and ask_for_api_key=True,
    the script will ask the user to paste the API key when running.
    """

    backend = backend.lower().strip()

    # -----------------------
    # Local LM Studio backend
    # -----------------------
    if backend == "local":
        if local_base_url is None or local_base_url.strip() == "":
            raise ValueError(
                "local_base_url is required for backend='local'. "
                "Example: local_base_url='http://localhost:1234'"
            )

        base_url = local_base_url.rstrip("/")
        headers = {}

        model = local_model

        if model is None and auto_detect_local_model:
            model = get_first_available_model(base_url, headers=headers)

        if model is None:
            raise ValueError(
                "No local model detected. "
                "Load a model in LM Studio or pass local_model='your-model-name'."
            )

        return BackendConfig(
            backend_name="local",
            base_url=base_url,
            chat_url=build_chat_url(base_url),
            model=model,
            headers=headers,
            timeout=timeout
        )

    # -----------------------
    # Generic API backend
    # -----------------------
    if backend == "api":
        if api_base_url is None:
            raise ValueError(
                "api_base_url is required for backend='api'. "
                "Example: api_base_url='https://api.groq.com/openai'"
            )

        if api_model is None:
            raise ValueError(
                "api_model is required for backend='api'. "
                "Example: api_model='llama-3.3-70b-versatile'"
            )

        if api_key is None and ask_for_api_key:
            api_key = input("Enter API key: ")

        if api_key is None:
            raise ValueError(
                "API key is missing. Provide api_key='your_key' "
                "or set ask_for_api_key=True."
            )

        auth_type = auth_type.lower().strip()

        if auth_type == "bearer":
            headers = {
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json"
            }

        elif auth_type == "api-key":
            headers = {
                "api-key": api_key,
                "Content-Type": "application/json"
            }

        else:
            raise ValueError("auth_type must be either 'bearer' or 'api-key'.")

        base_url = api_base_url.rstrip("/")

        return BackendConfig(
            backend_name="api",
            base_url=base_url,
            chat_url=build_chat_url(base_url),
            model=api_model,
            headers=headers,
            timeout=timeout
        )

    raise ValueError("backend must be either 'local' or 'api'.")



# =======================
# QUICK TEST
# =======================

if __name__ == "__main__":
    # Local LM Studio test
    config = get_backend_config(
        backend="local",
        local_base_url="http://localhost:1234"
    )

    print("Config loaded:")
    print(f"Backend: {config.backend_name}")
    print(f"Model: {config.model}")
    print(f"Chat URL: {config.chat_url}")
