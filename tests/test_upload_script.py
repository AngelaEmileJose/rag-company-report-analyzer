
import sys
import os
from pathlib import Path
from unittest.mock import MagicMock, patch
from fastapi.testclient import TestClient

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Mock key dependencies BEFORE importing api_server
# This prevents loading heavy libraries like sentence_transformers or torch
sys.modules["rag_system"] = MagicMock()
sys.modules["rag_system.ImprovedRAGSystem"] = MagicMock()
sys.modules["rag_system.RAGConfig"] = MagicMock()
sys.modules["rag_system.HealthCheck"] = MagicMock()
sys.modules["rag_system.PerformanceMetrics"] = MagicMock()

# Now import app
try:
    from api_server import app
    
    # We need to ensure the mocked rag_system in api_server behaves as expected
    # The api_server instantiates:
    # rag_system = ImprovedRAGSystem(config)
    # So we need to configure that instance
    
    # Get the mock objects that were injected
    import api_server
    mock_rag_instance = api_server.rag_system
    
    # Setup the async mock for process_document
    async def mock_process(source):
        class Result:
            success = True
            chunk_count = 10
            processing_time = 1.5
            source = source  # It should return the source passed to it
            message = "Success"
        return Result()

    mock_rag_instance.process_document.side_effect = mock_process

    def test_upload_flow():
        print("Starting Test...")
        client = TestClient(app)
        
        # Create dummy PDF content
        pdf_content = b"%PDF-1.4 header dummy content"
        filename = "test_doc.pdf"
        
        # Test upload
        print(f"Uploading {filename}...")
        response = client.post(
            "/upload",
            files={"file": (filename, pdf_content, "application/pdf")}
        )
        
        print(f"Status Code: {response.status_code}")
        print(f"Response: {response.json()}")
        
        assert response.status_code == 200
        assert response.json()["status"] == "success"
        assert response.json()["filename"] == filename
        # Check if source contains the filename (it should be a path)
        assert filename in response.json()["source"]
        
        print("Test PASSED!")

    if __name__ == "__main__":
        try:
            test_upload_flow()
        except Exception as e:
            print(f"Test FAILED: {e}")
            import traceback
            traceback.print_exc()

except ImportError as e:
    print(f"Import Error (likely missing fastapi/pydantic): {e}")
