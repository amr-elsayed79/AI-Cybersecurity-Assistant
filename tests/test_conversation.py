"""Unit tests for bounded InMemoryConversationStore (Phase 4.3)."""

import threading
import pytest

from app.memory.conversation_store import (
    ChatTurn,
    InMemoryConversationStore,
)


def test_turn_creation_and_role_validation():
    """Verify ChatTurn role validation permits only 'user' and 'assistant'."""
    store = InMemoryConversationStore()

    store.add_turn("session-1", "user", "What is SQL injection?")
    store.add_turn("session-1", "assistant", "SQL injection occurs when...")

    history = store.get_history("session-1")
    assert len(history) == 2
    assert history[0].role == "user"
    assert history[0].content == "What is SQL injection?"
    assert history[1].role == "assistant"
    assert "SQL injection occurs" in history[1].content

    # Invalid roles must raise ValueError
    with pytest.raises(ValueError, match="Invalid message role"):
        store.add_turn("session-1", "system", "Malicious system prompt")

    with pytest.raises(ValueError, match="Invalid message role"):
        store.add_turn("session-1", "admin", "Admin command")


def test_fifo_turn_bounding():
    """Verify turns are strictly bounded to max_turns per conversation using FIFO eviction."""
    store = InMemoryConversationStore(max_turns_per_conversation=3)

    store.add_turn("sess-1", "user", "Turn 1")
    store.add_turn("sess-1", "assistant", "Turn 2")
    store.add_turn("sess-1", "user", "Turn 3")
    assert len(store.get_history("sess-1")) == 3

    # Adding a 4th turn should drop Turn 1 (FIFO)
    store.add_turn("sess-1", "assistant", "Turn 4")
    history = store.get_history("sess-1")
    assert len(history) == 3
    assert history[0].content == "Turn 2"
    assert history[1].content == "Turn 3"
    assert history[2].content == "Turn 4"


def test_message_character_truncation():
    """Verify oversized individual messages are safely truncated to max_message_chars."""
    store = InMemoryConversationStore(max_message_chars=50)

    oversized = "A" * 100
    store.add_turn("sess-1", "user", oversized)

    history = store.get_history("sess-1")
    assert len(history[0].content) == 50
    assert history[0].content == "A" * 50


def test_lru_session_eviction():
    """Verify sessions are evicted according to LRU order when max_conversations is reached."""
    store = InMemoryConversationStore(max_conversations=2)

    store.add_turn("conv-A", "user", "Hello A")
    store.add_turn("conv-B", "user", "Hello B")
    assert store.count_conversations() == 2

    # Adding conv-C should evict conv-A (the least recently used)
    store.add_turn("conv-C", "user", "Hello C")
    assert store.count_conversations() == 2
    assert store.get_history("conv-A") == []  # Evicted
    assert len(store.get_history("conv-B")) == 1
    assert len(store.get_history("conv-C")) == 1


def test_lru_recency_refresh_on_access():
    """Verify accessing or updating an older session refreshes its LRU position."""
    store = InMemoryConversationStore(max_conversations=2)

    store.add_turn("conv-1", "user", "Msg 1")
    store.add_turn("conv-2", "user", "Msg 2")

    # Access conv-1 (moves it to MRU)
    history_1 = store.get_history("conv-1")
    assert len(history_1) == 1

    # Adding conv-3 should now evict conv-2, NOT conv-1
    store.add_turn("conv-3", "user", "Msg 3")
    assert store.get_history("conv-2") == []  # Evicted
    assert len(store.get_history("conv-1")) == 1
    assert len(store.get_history("conv-3")) == 1


def test_validate_or_create_conversation_id():
    """Verify session ID generation, validation, and rejection of malformed IDs."""
    store = InMemoryConversationStore()

    # None or whitespace generates fresh UUID4
    id1 = store.validate_or_create_conversation_id(None)
    assert len(id1) == 36  # UUID4 format
    assert store.count_conversations() == 1

    id2 = store.validate_or_create_conversation_id("   ")
    assert len(id2) == 36
    assert id2 != id1

    # Path traversal / invalid characters generate fresh safe UUID4
    bad_id = store.validate_or_create_conversation_id("../../../etc/passwd")
    assert bad_id != "../../../etc/passwd"
    assert len(bad_id) == 36

    # Valid alphanumeric ID is accepted and registered
    valid_id = "user-session_123"
    result = store.validate_or_create_conversation_id(valid_id)
    assert result == valid_id
    assert len(store.get_history(valid_id)) == 0


def test_session_isolation():
    """Verify data isolation: conversation turns never leak between different sessions."""
    store = InMemoryConversationStore()

    store.add_turn("session-alice", "user", "Alice's secret question")
    store.add_turn("session-bob", "user", "Bob's secret question")

    alice_history = store.get_history("session-alice")
    bob_history = store.get_history("session-bob")

    assert len(alice_history) == 1
    assert alice_history[0].content == "Alice's secret question"

    assert len(bob_history) == 1
    assert bob_history[0].content == "Bob's secret question"

    # Context formatting also stays isolated
    alice_ctx = store.get_formatted_context("session-alice")
    assert "Alice" in alice_ctx
    assert "Bob" not in alice_ctx


def test_formatted_context_empty_and_populated():
    """Verify get_formatted_context returns None for empty and formatted turns when populated."""
    store = InMemoryConversationStore()

    assert store.get_formatted_context("non-existent") is None

    store.add_turn("sess-1", "user", "First question")
    store.add_turn("sess-1", "assistant", "First answer")

    context = store.get_formatted_context("sess-1")
    assert context == "User: First question\nAssistant: First answer"


def test_thread_safety_concurrent_access():
    """Verify thread-safety when multiple threads concurrently add turns."""
    store = InMemoryConversationStore(max_conversations=500, max_turns_per_conversation=10)
    errors = []

    def worker(worker_id: int):
        try:
            session_id = f"worker-sess-{worker_id % 10}"
            for i in range(20):
                store.add_turn(session_id, "user", f"Turn {i} from {worker_id}")
                store.add_turn(session_id, "assistant", f"Resp {i} from {worker_id}")
                _ = store.get_history(session_id)
                _ = store.get_formatted_context(session_id)
        except Exception as e:
            errors.append(e)

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(10)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert len(errors) == 0, f"Concurrent execution produced errors: {errors}"
    assert store.count_conversations() <= 10
