from semantic_kernel.functions import kernel_function
from typing import Annotated

class RetrievalPlugin:
    """A native Semantic Kernel plugin that encapsulates the RAG retrieval logic."""
    
    def __init__(self, vector_store, retrieval_k: int = 5):
        self.vector_store = vector_store
        self.retrieval_k = retrieval_k

    @kernel_function(
        name="RetrieveContext",
        description="Searches the local FAISS vector store for passages relevant to the query."
    )
    def retrieve_context(
        self,
        query: Annotated[str, "The query to find relevant context for in the document"]
    ) -> str:
        """Query the vector store and return concatenated text chunks."""
        if not self.vector_store:
            return "Error: No vector store index has been initialized."
        
        try:
            # Retrieve documents from FAISS store
            docs = self.vector_store.similarity_search(query, k=self.retrieval_k)
            if not docs:
                return "No relevant information found in the document."
            
            # Combine content of retrieved chunks
            context = "\n\n".join([doc.page_content for doc in docs])
            return context
        except Exception as e:
            return f"Error during retrieval: {str(e)}"
