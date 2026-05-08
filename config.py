"""Shared configuration — loads .env and validates required environment variables."""

import os
import sys
from pathlib import Path
from dotenv import load_dotenv

# Load .env from the project root
load_dotenv(Path(__file__).parent / ".env")

REQUIRED_VARS = [
    "OPENAI_API_KEY",
    "OPENAI_BASE_URL",
    "LANGSMITH_API_KEY",
    "LANGSMITH_PROJECT",
]

OPTIONAL_VARS = {
    "LLM_MODEL": "gpt-4o-mini",
    "EMBEDDING_MODEL": "text-embedding-3-small",
}


def validate() -> dict:
    """Check required env vars exist. Returns dict with all config values."""
    missing = [v for v in REQUIRED_VARS if not os.environ.get(v)]
    if missing:
        print(f"❌ Missing environment variables: {', '.join(missing)}")
        print("   Please add them to your .env file.")
        sys.exit(1)

    config = {}
    for var in REQUIRED_VARS:
        config[var] = os.environ[var]
    for var, default in OPTIONAL_VARS.items():
        config[var] = os.environ.get(var, default)

    return config


def print_status(config: dict):
    """Print configuration status."""
    print("✅ Config loaded successfully")
    print(f"   LangSmith project : {config['LANGSMITH_PROJECT']}")
    print(f"   OpenAI endpoint   : {config['OPENAI_BASE_URL']}")
    print(f"   Default LLM model : {config['LLM_MODEL']}")
    print(f"   Embedding model   : {config['EMBEDDING_MODEL']}")


if __name__ == "__main__":
    cfg = validate()
    print_status(cfg)
