"""
Retrieval-Augmented Generation engine.
Allows the AI to answer questions about YOUR documents.
"""
import hashlib
import os
from pathlib import Path
from typing import Optional, List
from app.utils.logger import get_logger

logger = get_logger(__name__)

class RAGEngine:
    """
    Ingests documents and retrieves relevant context for queries.
    Uses ChromaDB for vector storage and sentence-transformers for embeddings.
    """
    
    def __init__(self, collection_name: str = "meta_ai_docs"):
        try:
            import chromadb
            from chromadb.config import Settings
        except ImportError:
            logger.error("chromadb not installed. RAG will be disabled.")
            self.collection = None
            return

        persist_dir = os.path.join("data", "vectordb")
        os.makedirs(persist_dir, exist_ok=True)
        
        self.client = chromadb.PersistentClient(
            path=persist_dir,
            settings=Settings(anonymized_telemetry=False)
        )
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"}
        )
        logger.info(
            f"RAG engine initialized. "
            f"Collection '{collection_name}' has "
            f"{self.collection.count()} documents."
        )
    
    def ingest_document(
        self, 
        file_path: str, 
        chunk_size: int = 500,
        chunk_overlap: int = 50
    ) -> int:
        """Load and index a document. Returns number of chunks created."""
        if not self.collection:
            return 0
            
        path = Path(file_path)
        text = self._extract_text(path)
        chunks = self._chunk_text(text, chunk_size, chunk_overlap)
        
        doc_id = hashlib.sha256(path.name.encode()).hexdigest()
        ids = [f"{doc_id}_{i}" for i in range(len(chunks))]
        metadatas = [
            {"source": path.name, "chunk": i, "total_chunks": len(chunks)} 
            for i in range(len(chunks))
        ]
        
        self.collection.upsert(
            documents=chunks,
            ids=ids,
            metadatas=metadatas
        )
        logger.info(f"Ingested '{path.name}': {len(chunks)} chunks indexed.")
        return len(chunks)
    
    def query(
        self, 
        question: str, 
        top_k: int = 5,
        score_threshold: float = 0.7
    ) -> Optional[str]:
        """Retrieve relevant context for a question."""
        if not self.collection or self.collection.count() == 0:
            return None
        
        results = self.collection.query(
            query_texts=[question],
            n_results=min(top_k, self.collection.count()),
            include=["documents", "distances", "metadatas"]
        )
        
        if not results["documents"] or not results["documents"][0]:
            return None
        
        # Filter by relevance score (cosine dist: lower = more similar)
        relevant = [
            (doc, meta, dist)
            for doc, meta, dist in zip(
                results["documents"][0],
                results["metadatas"][0],
                results["distances"][0]
            )
            if dist <= (1 - score_threshold)
        ]
        
        if not relevant:
            return None
        
        # Build context block
        context_parts = []
        for doc, meta, dist in relevant:
            source = meta.get("source", "unknown")
            relevance = round((1 - dist) * 100, 1)
            context_parts.append(
                f"[Source: {source} | Relevance: {relevance}%]\n{doc}"
            )
        
        return "\n\n---\n\n".join(context_parts)
    
    def _extract_text(self, path: Path) -> str:
        """Extract text from PDF, DOCX, or TXT files."""
        suffix = path.suffix.lower()
        
        if suffix == ".pdf":
            try:
                import fitz  # PyMuPDF
                doc = fitz.open(str(path))
                return "\n".join(page.get_text() for page in doc)
            except ImportError:
                return "PyMuPDF (fitz) not installed. Cannot extract PDF."
        
        elif suffix in [".docx", ".doc"]:
            try:
                import docx
                doc = docx.Document(str(path))
                return "\n".join(p.text for p in doc.paragraphs if p.text.strip())
            except ImportError:
                return "python-docx not installed. Cannot extract DOCX."
        
        elif suffix in [".txt", ".md", ".rst"]:
            return path.read_text(encoding="utf-8")
        
        elif suffix == ".csv":
            try:
                import pandas as pd
                df = pd.read_csv(path)
                return df.to_string()
            except ImportError:
                return "pandas not installed. Cannot extract CSV."
        
        else:
            return f"Unsupported file type: {suffix}"
    
    def _chunk_text(
        self, 
        text: str, 
        chunk_size: int, 
        overlap: int
    ) -> List[str]:
        """Split text into overlapping chunks for better context retrieval."""
        words = text.split()
        chunks = []
        i = 0
        while i < len(words):
            chunk = " ".join(words[i:i + chunk_size])
            if chunk.strip():
                chunks.append(chunk)
            i += chunk_size - overlap
        return chunks
