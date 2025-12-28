# fix_deprecations.py
import re
from pathlib import Path

def fix_pydantic_validator():
    """Fix Pydantic V1 @validator to V2 @field_validator"""
    rag_file = Path("RAG.py")

    with open(rag_file, 'r', encoding='utf-8') as f:
        content = f.read()

    # Replace import
    content = re.sub(
        r'from pydantic import BaseModel, Field, validator',
        r'from pydantic import BaseModel, Field, field_validator',
        content
    )

    # Replace decorator
    content = re.sub(
        r"@validator\('data_dir', 'cache_dir', 'vector_store_dir', 'logs_dir'\)",
        r"@field_validator('data_dir', 'cache_dir', 'vector_store_dir', 'logs_dir')",
        content
    )

    # Update method signature
    content = re.sub(
        r'def create_directories\(cls, v\):',
        r'def create_directories(cls, v):',
        content
    )

    with open(rag_file, 'w', encoding='utf-8') as f:
        f.write(content)

    print("✅ Fixed Pydantic validator deprecation")

def fix_embeddings():
    """Fix HuggingFace embeddings deprecation"""
    rag_file = Path("RAG.py")

    with open(rag_file, 'r', encoding='utf-8') as f:
        content = f.read()

    # Replace import
    content = re.sub(
        r'from langchain_community.embeddings import SentenceTransformerEmbeddings',
        r'from langchain_huggingface import HuggingFaceEmbeddings',
        content
    )

    # Replace usage
    content = re.sub(
        r'self.embeddings = SentenceTransformerEmbeddings\(',
        r'self.embeddings = HuggingFaceEmbeddings(',
        content
    )

    with open(rag_file, 'w', encoding='utf-8') as f:
        f.write(content)

    print("✅ Fixed HuggingFace embeddings deprecation")

if __name__ == "__main__":
    print("Fixing deprecation warnings...")
    fix_pydantic_validator()
    fix_embeddings()
    print("Done! You may need to install: pip install langchain-huggingface")