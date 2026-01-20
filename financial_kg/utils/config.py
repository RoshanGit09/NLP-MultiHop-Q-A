"""
Configuration management for FinancialKG
"""
import os
from pathlib import Path
from typing import Optional
from dotenv import load_dotenv
from pydantic import BaseModel, Field

# Load environment variables
env_path = Path(__file__).parent.parent / "config" / ".env"
load_dotenv(dotenv_path=env_path)


class GeminiConfig(BaseModel):
    """Gemini API configuration"""
    api_key: str = Field(default_factory=lambda: os.getenv("GEMINI_API_KEY", ""))
    model_fast: str = Field(default_factory=lambda: os.getenv("GEMINI_MODEL_FAST", "gemini-2.0-flash-exp"))
    model_pro: str = Field(default_factory=lambda: os.getenv("GEMINI_MODEL_PRO", "gemini-1.5-pro-latest"))
    embedding_model: str = Field(default_factory=lambda: os.getenv("GEMINI_EMBEDDING_MODEL", "models/text-embedding-004"))
    temperature: float = Field(default_factory=lambda: float(os.getenv("TEMPERATURE", "0.0")))
    max_retries: int = Field(default_factory=lambda: int(os.getenv("MAX_RETRIES", "3")))


class Neo4jConfig(BaseModel):
    """Neo4j database configuration"""
    uri: str = Field(default_factory=lambda: os.getenv("NEO4J_URI", "bolt://localhost:7687"))
    username: str = Field(default_factory=lambda: os.getenv("NEO4J_USERNAME", "neo4j"))
    password: str = Field(default_factory=lambda: os.getenv("NEO4J_PASSWORD", ""))


class DataConfig(BaseModel):
    """Data paths configuration"""
    data_dir: Path = Field(default_factory=lambda: Path(os.getenv("DATA_DIR", "./data")))
    cache_dir: Path = Field(default_factory=lambda: Path(os.getenv("CACHE_DIR", "./cache")))
    output_dir: Path = Field(default_factory=lambda: Path(os.getenv("OUTPUT_DIR", "./output")))
    
    def __init__(self, **data):
        super().__init__(**data)
        # Create directories if they don't exist
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.output_dir.mkdir(parents=True, exist_ok=True)


class ProcessingConfig(BaseModel):
    """Processing configuration"""
    max_workers: int = Field(default_factory=lambda: int(os.getenv("MAX_WORKERS", "8")))
    batch_size: int = Field(default_factory=lambda: int(os.getenv("BATCH_SIZE", "40")))
    chunk_size: int = Field(default_factory=lambda: int(os.getenv("CHUNK_SIZE", "400")))
    
    # Thresholds
    entity_threshold: float = Field(default_factory=lambda: float(os.getenv("ENTITY_THRESHOLD", "0.8")))
    relation_threshold: float = Field(default_factory=lambda: float(os.getenv("RELATION_THRESHOLD", "0.7")))
    entity_name_weight: float = Field(default_factory=lambda: float(os.getenv("ENTITY_NAME_WEIGHT", "0.8")))
    entity_label_weight: float = Field(default_factory=lambda: float(os.getenv("ENTITY_LABEL_WEIGHT", "0.2")))


class RealtimeConfig(BaseModel):
    """Real-time update configuration"""
    update_interval_minutes: int = Field(default_factory=lambda: int(os.getenv("UPDATE_INTERVAL_MINUTES", "15")))
    enable_realtime: bool = Field(default_factory=lambda: os.getenv("ENABLE_REALTIME", "false").lower() == "true")


class Config(BaseModel):
    """Main configuration aggregator"""
    gemini: GeminiConfig = Field(default_factory=GeminiConfig)
    neo4j: Neo4jConfig = Field(default_factory=Neo4jConfig)
    data: DataConfig = Field(default_factory=DataConfig)
    processing: ProcessingConfig = Field(default_factory=ProcessingConfig)
    realtime: RealtimeConfig = Field(default_factory=RealtimeConfig)
    
    # Logging
    log_level: str = Field(default_factory=lambda: os.getenv("LOG_LEVEL", "INFO"))
    log_file: Optional[str] = Field(default_factory=lambda: os.getenv("LOG_FILE"))


# Global config instance
config = Config()


def get_config() -> Config:
    """Get the global configuration instance"""
    return config


if __name__ == "__main__":
    # Test configuration
    cfg = get_config()
    print("Configuration loaded successfully!")
    print(f"Gemini Fast Model: {cfg.gemini.model_fast}")
    print(f"Gemini Pro Model: {cfg.gemini.model_pro}")
    print(f"Neo4j URI: {cfg.neo4j.uri}")
    print(f"Data Directory: {cfg.data.data_dir}")
    print(f"Max Workers: {cfg.processing.max_workers}")
