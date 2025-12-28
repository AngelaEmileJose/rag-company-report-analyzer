import os
import yaml
from dotenv import load_dotenv
from pathlib import Path
from typing import Optional
import sys
from pydantic import BaseModel, Field, field_validator, model_validator, ValidationError
from .models import LogLevel

class RAGConfig(BaseModel):
    """Unified configuration for RAG system"""
    
    # API Configuration
    groq_api_key: str = Field(..., description="Groq API key")
    model_name: str = Field(default="llama-3.3-70b-versatile")
    temperature: float = Field(default=0.1, ge=0.0, le=2.0)
    max_tokens: Optional[int] = Field(default=None, ge=1, le=8192)

    # Processing Configuration
    chunk_size: int = Field(default=1000, ge=100, le=4000)
    chunk_overlap: int = Field(default=200, ge=0, le=1000)
    max_file_size_mb: int = Field(default=50, ge=1, le=500)

    # Vector Store Configuration
    embedding_model: str = Field(default="sentence-transformers/all-MiniLM-L6-v2")
    vector_store_type: str = Field(default="faiss", pattern="^(faiss|chroma|pinecone)$")
    retrieval_k: int = Field(default=8, ge=1, le=20)
    score_threshold: float = Field(default=0.0, ge=0.0, le=1.0)

    # Caching Configuration
    enable_cache: bool = Field(default=True)
    cache_ttl_hours: int = Field(default=24, ge=1, le=168)
    redis_url: Optional[str] = Field(default=None)

    # File System Configuration
    data_dir: Path = Field(default=Path("./data"))
    cache_dir: Path = Field(default=Path("./cache"))
    vector_store_dir: Path = Field(default=Path("./vector_stores"))
    logs_dir: Path = Field(default=Path("./logs"))

    # Performance Configuration
    max_concurrent_downloads: int = Field(default=3, ge=1, le=10)
    request_timeout: int = Field(default=60, ge=10, le=300)
    max_retries: int = Field(default=3, ge=0, le=10)

    # Logging Configuration
    log_level: LogLevel = Field(default=LogLevel.INFO)
    log_to_file: bool = Field(default=True)
    max_log_size_mb: int = Field(default=10, ge=1, le=100)

    # Security Configuration
    allowed_file_types: list = Field(default=[".pdf", ".txt", ".docx"])
    max_url_redirects: int = Field(default=5, ge=0, le=10)

    # Debug
    enable_debug: bool = Field(default=False)

    @model_validator(mode='after')
    def create_directories(self) -> 'RAGConfig':
        """Ensure directories exist"""
        for directory in [self.data_dir, self.cache_dir, self.vector_store_dir, self.logs_dir]:
            directory.mkdir(parents=True, exist_ok=True)
        return self

    @classmethod
    def from_yaml(cls, config_path: Path) -> 'RAGConfig':
        """Load configuration from YAML file"""
        with open(config_path, 'r') as f:
            config_data = yaml.safe_load(f)
        return cls(**config_data)

    @classmethod
    def from_env(cls) -> 'RAGConfig':
        """Load configuration from environment variables"""
        load_dotenv()
        config_data = {}

        env_mapping = {
            'GROQ_API_KEY': 'groq_api_key',
            'MODEL_NAME': 'model_name',
            'TEMPERATURE': 'temperature',
            'CHUNK_SIZE': 'chunk_size',
            'CHUNK_OVERLAP': 'chunk_overlap',
            'MAX_FILE_SIZE_MB': 'max_file_size_mb',
            'EMBEDDING_MODEL': 'embedding_model',
            'ENABLE_CACHE': 'enable_cache',
            'CACHE_TTL_HOURS': 'cache_ttl_hours',
            'REDIS_URL': 'redis_url',
            'LOG_LEVEL': 'log_level',
        }

        for env_var, config_field in env_mapping.items():
            if env_value := os.getenv(env_var):
                if config_field in ['temperature', 'score_threshold']:
                    config_data[config_field] = float(env_value)
                elif config_field in ['chunk_size', 'chunk_overlap', 'max_file_size_mb', 'cache_ttl_hours']:
                    config_data[config_field] = int(env_value)
                elif config_field in ['enable_cache', 'log_to_file', 'enable_debug']:
                    config_data[config_field] = env_value.lower() in ['true', '1', 'yes']
                else:
                    config_data[config_field] = env_value

        try:
            return cls(**config_data)
        except ValidationError as e:
            print("\n" + "="*60)
            print("❌ MISSING CONFIGURATION: GROQ_API_KEY")
            print("="*60)
            print("The application cannot start because the API Key is missing.")
            print("\nPLEASE FOLLOW THESE STEPS:")
            print("1. Create a file named '.env' in this folder: " + str(Path.cwd()))
            print("2. Open the file and add your key like this:")
            print("   GROQ_API_KEY=gsk_your_key_here")
            print("\n(See .env.example for a template)")
            print("="*60 + "\n")
            sys.exit(1)
