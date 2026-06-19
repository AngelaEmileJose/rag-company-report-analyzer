"""Tests for the Semantic Kernel + Ollama subproject.

These tests target the parts that can be verified without spinning up Ollama
or downloading the sentence-transformers model:

  - RetrievalPlugin output format (structured context with [Chunk N | Page P | score S] headers)
  - RetrievalPlugin error paths
  - Plugin folder layout (the four semantic skills are present and well-formed)
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import List, Tuple

try:
    import pytest  # type: ignore
    _HAS_PYTEST = True
except ImportError:  # allow running this file as a plain script when pytest is not installed
    pytest = None  # type: ignore
    _HAS_PYTEST = False


# --- test doubles ----------------------------------------------------------

class _StubDocument:
    """Minimal stand-in for langchain_core.documents.Document."""
    def __init__(self, page_content: str, metadata: dict | None = None):
        self.page_content = page_content
        self.metadata = metadata or {}


class _StubVectorStore:
    """Stand-in for langchain_community.vectorstores.FAISS that returns Documents with metadata."""
    def __init__(self, hits: List[Tuple[_StubDocument, float]]):
        self._hits = hits
        self.last_query: str | None = None
        self.last_k: int | None = None

    def similarity_search_with_score(self, query: str, k: int):
        self.last_query = query
        self.last_k = k
        return self._hits[:k]


# --- imports under test ----------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from retrieval_plugin import RetrievalPlugin  # noqa: E402


# --- RetrievalPlugin tests --------------------------------------------------

def test_retrieval_plugin_emits_structured_context_with_chunk_and_page():
    docs = [
        (_StubDocument("Revenue grew 8% YoY to $25.2B.", {"page": 3}), 0.21),
        (_StubDocument("Operating margin was 10.8%.", {"page": 4}), 0.34),
    ]
    plugin = RetrievalPlugin(vector_store=_StubVectorStore(docs), retrieval_k=5)

    out = plugin.retrieve_context("financial performance")

    # Both chunks present, indexed 1 and 2, with page numbers and scores.
    assert "[Chunk 1 | Page 3 | score 0.2100]" in out
    assert "[Chunk 2 | Page 4 | score 0.3400]" in out
    # The chunk text must follow the header, separated by --- between blocks.
    assert "Revenue grew 8% YoY to $25.2B." in out
    assert "Operating margin was 10.8%." in out
    assert "\n\n---\n\n" in out


def test_retrieval_plugin_respects_retrieval_k():
    docs = [
        (_StubDocument(f"chunk {i} text", {"page": i + 1}), 0.1 * (i + 1))
        for i in range(10)
    ]
    plugin = RetrievalPlugin(vector_store=_StubVectorStore(docs), retrieval_k=3)

    out = plugin.retrieve_context("anything")

    # Only 3 chunks should appear.
    assert "[Chunk 1 | Page 1 |" in out
    assert "[Chunk 2 | Page 2 |" in out
    assert "[Chunk 3 | Page 3 |" in out
    assert "[Chunk 4" not in out


def test_retrieval_plugin_falls_back_to_unknown_page_when_metadata_missing():
    """Documents without a 'page' key in metadata should not crash; header shows 'unknown'."""
    docs = [(_StubDocument("orphan chunk", {}), 0.5)]
    plugin = RetrievalPlugin(vector_store=_StubVectorStore(docs), retrieval_k=1)

    out = plugin.retrieve_context("x")

    assert "[Chunk 1 | Page unknown | score 0.5000]" in out
    assert "orphan chunk" in out


def test_retrieval_plugin_can_omit_scores():
    docs = [(_StubDocument("text", {"page": 2}), 0.99)]
    plugin = RetrievalPlugin(vector_store=_StubVectorStore(docs), retrieval_k=1, include_scores=False)

    out = plugin.retrieve_context("q")

    assert "[Chunk 1 | Page 2]" in out
    assert "score" not in out


def test_retrieval_plugin_returns_error_when_no_vector_store():
    plugin = RetrievalPlugin(vector_store=None, retrieval_k=5)

    out = plugin.retrieve_context("anything")

    assert "Error: No vector store index has been initialized." in out


def test_retrieval_plugin_returns_empty_message_when_no_results():
    plugin = RetrievalPlugin(vector_store=_StubVectorStore([]), retrieval_k=5)

    out = plugin.retrieve_context("anything")

    assert out == "No relevant information found in the document."


def test_retrieval_plugin_swallows_vector_store_exceptions():
    class _BrokenStore:
        def similarity_search_with_score(self, query, k):
            raise RuntimeError("FAISS index corrupted")

    plugin = RetrievalPlugin(vector_store=_BrokenStore(), retrieval_k=5)

    out = plugin.retrieve_context("anything")

    assert "Error during retrieval: FAISS index corrupted" in out


# --- plugin folder layout tests -------------------------------------------

REPORT_PLUGIN_DIR = PROJECT_ROOT / "plugins" / "ReportPlugin"


@pytest.mark.parametrize(
    "skill_name",
    ["SummarizeReport", "ExtractKeyMetrics", "ExtractKeyMetricsV2", "AnswerWithCitation", "IdentifyRiskFactors"],
)
def test_semantic_skill_folder_layout(skill_name: str):
    skill_dir = REPORT_PLUGIN_DIR / skill_name
    assert skill_dir.is_dir(), f"Missing skill folder: {skill_dir}"

    config = skill_dir / "config.json"
    prompt = skill_dir / "skprompt.txt"
    assert config.is_file(), f"Missing config.json in {skill_dir}"
    assert prompt.is_file(), f"Missing skprompt.txt in {skill_dir}"
    assert prompt.read_text(encoding="utf-8").strip(), f"Empty prompt in {skill_dir}"


@pytest.mark.parametrize(
    "skill_name",
    ["ExtractKeyMetricsV2", "AnswerWithCitation", "IdentifyRiskFactors"],
)
def test_new_skill_config_validates_as_json_and_has_required_fields(skill_name: str):
    config = json.loads((REPORT_PLUGIN_DIR / skill_name / "config.json").read_text(encoding="utf-8"))
    assert config.get("schema") == 1
    assert "execution_settings" in config
    assert "default" in config["execution_settings"]
    assert "input_variables" in config
    for var in config["input_variables"]:
        assert "name" in var
        assert var.get("required") is True
