import pytest
from pathlib import Path
from rag_system import RAGConfig, ImprovedRAGSystem, DocumentProcessor

@pytest.fixture
def test_config():
    return RAGConfig(
        groq_api_key="test_key",
        enable_cache=False,
        chunk_size=100,
        chunk_overlap=10
    )

def test_config_initialization(test_config):
    assert test_config.groq_api_key == "test_key"
    assert test_config.chunk_size == 100

def test_rag_system_initialization(test_config):
    rag = ImprovedRAGSystem(test_config)
    assert rag.config == test_config
    assert rag.document_processor is not None

@pytest.mark.asyncio
async def test_document_processor_file_not_found(test_config):
    processor = DocumentProcessor(test_config)
    with pytest.raises(FileNotFoundError):
        await processor.process_source("non_existent_file.pdf")
