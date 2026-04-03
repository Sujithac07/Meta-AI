"""
Unit tests for RAG (Retrieval-Augmented Generation) engine.
"""
import tempfile
import os
from pathlib import Path
from unittest.mock import patch, MagicMock
import sys

# Add the project root to the path
sys.path.insert(0, str(Path(__file__).parent.parent))


class TestRAGEngine:
    """Tests for the RAG Engine functionality."""
    
    def test_rag_engine_init(self):
        """Test RAG engine initialization."""
        # This tests that the class can be instantiated
        # We'll mock chromadb to avoid actual file system operations
        with patch('chromadb.PersistentClient') as mock_client:
            from app.core.rag_engine import RAGEngine
            
            # Create a mock collection
            mock_collection = MagicMock()
            mock_collection.count.return_value = 0
            mock_client.return_value.get_or_create_collection.return_value = mock_collection
            
            rag = RAGEngine(collection_name="test_collection")
            
            assert rag.collection is not None
            mock_client.assert_called_once()
    
    def test_chunk_text_basic(self):
        """Test text chunking with basic input."""
        with patch('chromadb.PersistentClient'):
            from app.core.rag_engine import RAGEngine
            
            rag = RAGEngine.__new__(RAGEngine)  # Create instance without __init__
            rag.collection = MagicMock()
            
            text = "Hello world this is a test document"
            chunks = rag._chunk_text(text, chunk_size=5, overlap=1)
            
            # Should split into multiple chunks
            assert len(chunks) > 1
    
    def test_chunk_text_single_chunk(self):
        """Test text chunking when text is shorter than chunk size."""
        with patch('chromadb.PersistentClient'):
            from app.core.rag_engine import RAGEngine
            
            rag = RAGEngine.__new__(RAGEngine)
            rag.collection = MagicMock()
            
            text = "Short text"
            chunks = rag._chunk_text(text, chunk_size=500, overlap=50)
            
            assert len(chunks) == 1
            assert chunks[0] == "Short text"
    
    def test_chunk_text_with_overlap(self):
        """Test that chunks have proper overlap."""
        with patch('chromadb.PersistentClient'):
            from app.core.rag_engine import RAGEngine
            
            rag = RAGEngine.__new__(RAGEngine)
            rag.collection = MagicMock()
            
            # Create text that will produce multiple chunks
            text = " ".join([f"word{i}" for i in range(100)])
            chunks = rag._chunk_text(text, chunk_size=20, overlap=5)
            
            # With 100 words, chunk_size=20, overlap=5:
            # chunk 1: words 0-19
            # chunk 2: words 15-34 (overlap of 5)
            # So we should have more than 5 chunks
            assert len(chunks) >= 5
    
    def test_extract_text_txt(self):
        """Test text extraction from TXT files."""
        with patch('chromadb.PersistentClient'):
            from app.core.rag_engine import RAGEngine
            
            rag = RAGEngine.__new__(RAGEngine)
            rag.collection = MagicMock()
            
            # Create a temporary txt file
            with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
                f.write("This is a test document content.")
                temp_path = f.name
            
            try:
                text = rag._extract_text(Path(temp_path))
                assert "test document content" in text
            finally:
                os.unlink(temp_path)
    
    def test_extract_text_csv(self):
        """Test text extraction from CSV files."""
        with patch('chromadb.PersistentClient'):
            from app.core.rag_engine import RAGEngine
            
            rag = RAGEngine.__new__(RAGEngine)
            rag.collection = MagicMock()
            
            # Create a temporary CSV file
            with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
                f.write("name,age,city\n")
                f.write("Alice,30,NYC\n")
                f.write("Bob,25,LA\n")
                temp_path = f.name
            
            try:
                text = rag._extract_text(Path(temp_path))
                assert "Alice" in text
                assert "Bob" in text
            finally:
                os.unlink(temp_path)
    
    def test_extract_text_unsupported(self):
        """Test extraction with unsupported file type."""
        with patch('chromadb.PersistentClient'):
            from app.core.rag_engine import RAGEngine
            
            rag = RAGEngine.__new__(RAGEngine)
            rag.collection = MagicMock()
            
            result = rag._extract_text(Path("test.xyz"))
            assert "Unsupported file type" in result
    
    def test_query_no_documents(self):
        """Test query returns None when no documents indexed."""
        with patch('chromadb.PersistentClient') as mock_client:
            from app.core.rag_engine import RAGEngine
            
            mock_collection = MagicMock()
            mock_collection.count.return_value = 0
            mock_client.return_value.get_or_create_collection.return_value = mock_collection
            
            rag = RAGEngine()
            
            result = rag.query("test question")
            assert result is None
    
    def test_query_with_documents(self):
        """Test query returns context when documents are indexed."""
        with patch('chromadb.PersistentClient') as mock_client:
            from app.core.rag_engine import RAGEngine
            
            mock_collection = MagicMock()
            mock_collection.count.return_value = 5
            mock_collection.query.return_value = {
                "documents": [["Relevant document content here"]],
                "metadatas": [[{"source": "test.txt"}]],
                "distances": [[0.3]]
            }
            mock_client.return_value.get_or_create_collection.return_value = mock_collection
            
            rag = RAGEngine()
            
            result = rag.query("test question", top_k=3, score_threshold=0.7)
            
            assert result is not None
            assert "Relevant document content" in result
    
    def test_ingest_document(self):
        """Test document ingestion."""
        with patch('chromadb.PersistentClient') as mock_client:
            from app.core.rag_engine import RAGEngine
            
            mock_collection = MagicMock()
            mock_collection.count.return_value = 0
            mock_client.return_value.get_or_create_collection.return_value = mock_collection
            
            # Create a temporary txt file
            with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
                f.write("This is test document content for RAG testing.")
                temp_path = f.name
            
            try:
                rag = RAGEngine()
                chunks_created = rag.ingest_document(temp_path, chunk_size=10, chunk_overlap=2)
                
                # Verify upsert was called
                mock_collection.upsert.assert_called_once()
                assert chunks_created > 0
            finally:
                os.unlink(temp_path)


class TestRAGQueryFormatting:
    """Tests for RAG query result formatting."""
    
    def test_query_result_format(self):
        """Test that query results are properly formatted with source and relevance."""
        with patch('chromadb.PersistentClient') as mock_client:
            from app.core.rag_engine import RAGEngine
            
            mock_collection = MagicMock()
            mock_collection.count.return_value = 2
            mock_collection.query.return_value = {
                "documents": [["First doc content", "Second doc content"]],
                "metadatas": [[{"source": "doc1.txt"}, {"source": "doc2.txt"}]],
                "distances": [[0.2, 0.4]]  # Lower = more similar
            }
            mock_client.return_value.get_or_create_collection.return_value = mock_collection
            
            rag = RAGEngine()
            
            result = rag.query("test question", top_k=2, score_threshold=0.5)
            
            # Should include source and relevance
            assert "doc1.txt" in result
            assert "doc2.txt" in result
            assert "Relevance" in result
    
    def test_query_filters_by_threshold(self):
        """Test that query filters out irrelevant results."""
        with patch('chromadb.PersistentClient') as mock_client:
            from app.core.rag_engine import RAGEngine
            
            mock_collection = MagicMock()
            mock_collection.count.return_value = 2
            # First result is relevant (0.3 distance = 70% relevance)
            # Second result is not relevant (0.8 distance = 20% relevance)
            mock_collection.query.return_value = {
                "documents": [["Relevant", "Not relevant"]],
                "metadatas": [[{"source": "good.txt"}, {"source": "bad.txt"}]],
                "distances": [[0.3, 0.8]]
            }
            mock_client.return_value.get_or_create_collection.return_value = mock_collection
            
            rag = RAGEngine()
            
            # With 70% threshold, only results >= 30% distance should pass
            result = rag.query("test", top_k=2, score_threshold=0.7)
            
            # With score_threshold=0.7, distance must be <= 0.3
            # Only first result passes
            assert "Relevant" in result
            assert "Not relevant" not in result or result.count("Not relevant") == 0
