#!/usr/bin/env python3
"""
Setup and Run Script for RAG System
This script helps initialize and run the RAG system with proper configuration
"""

import os
import sys
from pathlib import Path
import subprocess
import json

def check_requirements():
    """Check if required packages are installed"""
    try:
        import langchain
        import groq
        import rich
        import click
        import aiofiles
        import aiohttp
        print("✓ All required packages found")
        return True
    except ImportError as e:
        print(f"✗ Missing required package: {e}")
        print("Please install requirements: pip install -r requirements.txt")
        return False

def create_directories():
    """Create necessary directories"""
    directories = [
        "config",
        "data",
        "cache",
        "vector_stores",
        "logs",
        "results"
    ]

    for dir_name in directories:
        Path(dir_name).mkdir(exist_ok=True)
        print(f"✓ Created directory: {dir_name}")

def check_api_key():
    """Check if GROQ API key is set"""
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        print("✗ GROQ_API_KEY environment variable not set")
        print("Please set your Groq API key:")
        print("  export GROQ_API_KEY=your_api_key_here")
        return False

    print("✓ GROQ_API_KEY found")
    return True

def create_config_file():
    """Create production config file"""
    config_path = Path("config/production.yaml")

    if config_path.exists():
        print("✓ Configuration file already exists")
        return

    config_content = '''# Production Configuration for RAG System
# API Configuration
groq_api_key: "${GROQ_API_KEY}"  # Will be loaded from environment
model_name: "llama3-70b-8192"
temperature: 0.1
max_tokens: null

# Processing Configuration  
chunk_size: 1000
chunk_overlap: 200
max_file_size_mb: 50

# Vector Store Configuration
embedding_model: "sentence-transformers/all-MiniLM-L6-v2"
vector_store_type: "faiss"
retrieval_k: 5
score_threshold: 0.0

# Caching Configuration
enable_cache: true
cache_ttl_hours: 24
redis_url: null

# File System Configuration
data_dir: "./data"
cache_dir: "./cache"  
vector_store_dir: "./vector_stores"
logs_dir: "./logs"

# Performance Configuration
max_concurrent_downloads: 3
request_timeout: 60
max_retries: 3

# Logging Configuration
log_level: "INFO"
log_to_file: true
max_log_size_mb: 10

# Security Configuration
allowed_file_types:
  - ".pdf"
  - ".txt" 
  - ".docx"
max_url_redirects: 5
'''

    with open(config_path, 'w') as f:
        f.write(config_content)

    print(f"✓ Created configuration file: {config_path}")

def main():
    print("RAG System Setup")
    print("================")

    # Check requirements
    if not check_requirements():
        sys.exit(1)

    # Create directories
    create_directories()

    # Check API key
    if not check_api_key():
        sys.exit(1)

    # Create config file
    create_config_file()

    print("\n✓ Setup complete!")
    print("\nUsage examples:")
    print("1. Interactive mode:")
    print("   python rag_question_cli_fixed.py interactive")
    print("\n2. Analyze document:")
    print("   python rag_question_cli_fixed.py analyze --url 'https://example.com/report.pdf' --company 'Apple Inc.' --topic 'sustainability' --count 5 --export json --output results.json")
    print("\n3. Analyze local file:")
    print("   python rag_question_cli_fixed.py analyze --file 'report.pdf' --company 'Company Name' --topic 'financial' --export txt --output results.txt")

if __name__ == "__main__":
    main()