from semantic_kernel.functions import kernel_function
from typing import Annotated


class RetrievalPlugin:
    """A native Semantic Kernel plugin that encapsulates the RAG retrieval logic.

    Returns structured context so downstream semantic skills can cite sources:
    each chunk is formatted as a block with chunk index, source page, and
    similarity score, followed by the chunk text.
    """

    def __init__(self, vector_store, retrieval_k: int = 5, include_scores: bool = True):
        self.vector_store = vector_store
        self.retrieval_k = retrieval_k
        self.include_scores = include_scores

    @kernel_function(
        name="RetrieveContext",
        description=(
            "Searches the local FAISS vector store for passages relevant to the query. "
            "Returns structured context where each chunk is prefixed with its index, "
            "source page number, and similarity score so downstream skills can cite sources."
        ),
    )
    def retrieve_context(
        self,
        query: Annotated[str, "The query to find relevant context for in the document"],
    ) -> str:
        """Query the vector store and return concatenated text chunks with provenance."""
        if not self.vector_store:
            return "Error: No vector store index has been initialized."

        try:
            # similarity_search_with_score returns (Document, score) tuples.
            # Lower score = more similar for FAISS L2 distance.
            results = self.vector_store.similarity_search_with_score(query, k=self.retrieval_k)
            if not results:
                return "No relevant information found in the document."

            blocks = []
            for idx, (doc, score) in enumerate(results, start=1):
                page = (doc.metadata or {}).get("page", "unknown")
                if self.include_scores:
                    header = f"[Chunk {idx} | Page {page} | score {score:.4f}]"
                else:
                    header = f"[Chunk {idx} | Page {page}]"
                blocks.append(f"{header}\n{doc.page_content}")

            return "\n\n---\n\n".join(blocks)
        except Exception as e:
            return f"Error during retrieval: {str(e)}"
