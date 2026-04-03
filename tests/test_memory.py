"""
Unit tests for conversation memory module.
"""
import pytest
import tempfile
from pathlib import Path
from unittest.mock import patch
import sys

# Add the project root to the path
sys.path.insert(0, str(Path(__file__).parent.parent))

from app.core.memory import ConversationMemory


@pytest.fixture
def temp_db():
    """Create a temporary database for testing."""
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = Path(tmpdir) / "test.db"
        with patch("app.core.memory.DB_PATH", db_path):
            yield db_path


def test_conversation_memory_init(temp_db):
    """Test that ConversationMemory initializes correctly."""
    with patch("app.core.memory.DB_PATH", temp_db):
        memory = ConversationMemory(session_id="test_session_001")
        assert memory.session_id == "test_session_001"
        assert memory.max_history == 20


def test_add_message(temp_db):
    """Test adding a message to the conversation."""
    with patch("app.core.memory.DB_PATH", temp_db):
        memory = ConversationMemory(session_id="test_session_001")
        memory.add_message("user", "Hello, AI!")
        
        history = memory.get_history()
        assert len(history) == 1
        assert history[0].role == "user"
        assert history[0].content == "Hello, AI!"


def test_add_multiple_messages(temp_db):
    """Test adding multiple messages."""
    with patch("app.core.memory.DB_PATH", temp_db):
        memory = ConversationMemory(session_id="test_session_001")
        
        memory.add_message("user", "Hello")
        memory.add_message("assistant", "Hi there!")
        memory.add_message("user", "How are you?")
        
        history = memory.get_history()
        assert len(history) == 3
        assert history[0].content == "Hello"
        assert history[1].content == "Hi there!"
        assert history[2].content == "How are you?"


def test_max_history_limit(temp_db):
    """Test that max_history limit is enforced."""
    with patch("app.core.memory.DB_PATH", temp_db):
        memory = ConversationMemory(session_id="test_session_001", max_history=5)
        
        for i in range(10):
            memory.add_message("user", f"Message {i}")
        
        history = memory.get_history()
        assert len(history) == 5


def test_clear_session(temp_db):
    """Test clearing a session."""
    with patch("app.core.memory.DB_PATH", temp_db):
        memory = ConversationMemory(session_id="test_session_001")
        memory.add_message("user", "This should be deleted")
        
        assert len(memory.get_history()) == 1
        
        memory.clear()
        assert len(memory.get_history()) == 0


def test_multiple_sessions_isolated(temp_db):
    """Test that different sessions are isolated."""
    with patch("app.core.memory.DB_PATH", temp_db):
        session_a = ConversationMemory("session_A")
        session_b = ConversationMemory("session_B")
        
        session_a.add_message("user", "Message in A")
        session_b.add_message("user", "Message in B")
        
        assert len(session_a.get_history()) == 1
        assert len(session_b.get_history()) == 1
        assert session_a.get_history()[0].content == "Message in A"
        assert session_b.get_history()[0].content == "Message in B"


def test_stats(temp_db):
    """Test getting session statistics."""
    with patch("app.core.memory.DB_PATH", temp_db):
        memory = ConversationMemory(session_id="test_session_001")
        memory.add_message("user", "Test")
        
        stats = memory.get_summary_stats()
        assert stats["total_messages"] == 1
        assert stats["first_message"] is not None
        assert stats["last_message"] is not None


def test_message_with_metadata(temp_db):
    """Test adding messages with metadata."""
    with patch("app.core.memory.DB_PATH", temp_db):
        memory = ConversationMemory(session_id="test_session_001")
        metadata = {"model": "gpt-4", "tokens": 150}
        
        memory.add_message("user", "Hello", metadata=metadata)
        
        # Verify message was added (metadata is stored but not returned in get_history)
        history = memory.get_history()
        assert len(history) == 1
        assert history[0].content == "Hello"


def test_chronological_order(temp_db):
    """Test that messages are returned in chronological order."""
    with patch("app.core.memory.DB_PATH", temp_db):
        memory = ConversationMemory(session_id="test_session_001")
        
        memory.add_message("user", "First")
        memory.add_message("assistant", "Second")
        memory.add_message("user", "Third")
        
        history = memory.get_history()
        assert history[0].content == "First"
        assert history[1].content == "Second"
        assert history[2].content == "Third"
