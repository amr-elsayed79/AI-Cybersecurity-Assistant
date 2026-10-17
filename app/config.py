"""Configuration settings for the AI Cybersecurity Assistant."""

import json
from pathlib import Path
from typing import List, Optional, Set, Union
from pydantic import field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application and pipeline configuration."""

    PROJECT_NAME: str = "AI Cybersecurity Assistant"
    VERSION: str = "0.2.0"

    # Base Paths
    BASE_DIR: Path = Path(__file__).resolve().parent.parent
    DATA_DIR: Path = BASE_DIR / "data"
    MASTER_KB_DIR: Path = DATA_DIR / "master"
    MASTER_KB_PATH: Path = MASTER_KB_DIR / "ai_security_vulnerability_kb_master_v0.2.json"

    # Knowledge Base Baseline Metrics
    BASELINE_KB_SHA256: str = "d59af97e75dce8540f5dfaf1ba2d4e4afbcfe99c2ac697779bb5e9c6c0e99471"
    EXPECTED_CANONICAL_RECORDS: int = 74
    EXPECTED_ACTIVE_SCENARIOS: int = 119

    # Affected Records with Verified Field Displacement
    AFFECTED_RECORD_IDS: Set[str] = {
        "sql-injection",
        "nosql-injection",
        "os-command-injection",
        "ssti",
        "xxe",
    }

    # Vector Storage and Embedding Configuration (Phase 3)
    EMBEDDING_MODEL_NAME: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    EMBEDDING_DIMENSION: int = 384
    EMBEDDING_DEVICE: str = "cpu"
    QDRANT_STORAGE_PATH: Path = DATA_DIR / "qdrant_db"
    QDRANT_COLLECTION_NAME: str = "cybersecurity_kb_chunks"
    TOP_K_RETRIEVAL: int = 5

    # Hybrid Retrieval Configuration (Phase 3 Verified)
    ENABLE_HYBRID_RETRIEVAL: bool = True
    HYBRID_BM25_WEIGHT: float = 0.15

    # LLM Provider Configuration (Phase 4)
    LLM_PROVIDER: str = "mock"
    GROQ_API_KEY: Optional[str] = None
    GROQ_MODEL: str = "llama-3.3-70b-versatile"
    GROQ_TEMPERATURE: float = 0.1
    GROQ_MAX_TOKENS: int = 2048
    GROQ_TIMEOUT_SECONDS: float = 20.0
    GROQ_MAX_RETRIES: int = 2

    # Experimental Retrieval Threshold (Disabled by default; does not alter Phase 3 retrieval)
    RETRIEVAL_SCORE_THRESHOLD: Optional[float] = None

    # API and Conversation Settings (Phase 4.3)
    API_V1_PREFIX: str = "/api/v1"
    MAX_CONVERSATIONS: int = 1000
    MAX_TURNS_PER_CONVERSATION: int = 6
    MAX_MESSAGE_CHARS: int = 2000
    CORS_ALLOW_ORIGINS: Union[List[str], str] = [
        "http://localhost:3000",
        "http://localhost:8000",
        "http://127.0.0.1:3000",
        "http://127.0.0.1:8000",
    ]

    @field_validator("CORS_ALLOW_ORIGINS", mode="after")
    @classmethod
    def parse_cors_allow_origins(cls, v: Union[List[str], str]) -> List[str]:
        """Support both JSON array and comma-separated origin strings in .env."""
        if isinstance(v, str):
            v_str = v.strip()
            if v_str.startswith("[") and v_str.endswith("]"):
                try:
                    parsed = json.loads(v_str)
                    if isinstance(parsed, list):
                        return [str(origin).strip() for origin in parsed if str(origin).strip()]
                except Exception:
                    pass
            return [origin.strip() for origin in v_str.split(",") if origin.strip()]
        return [str(origin).strip() for origin in v if str(origin).strip()]

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )


settings = Settings()
