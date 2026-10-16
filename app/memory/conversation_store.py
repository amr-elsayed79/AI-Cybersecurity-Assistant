"""In-memory bounded conversation storage for multi-turn chat sessions.

IMPORTANT ARCHITECTURAL & DURABILITY DISCLOSURE:
This in-memory store is ephemeral, bounded, and designed strictly for single-process
local execution and demonstrations.
- Durability: ALL conversation histories are volatile and lost immediately on process restart.
- Concurrency: This store is NOT shared across multiple Uvicorn worker processes (workers > 1).
- Network Binding: The API must remain bound to 127.0.0.1. Do NOT expose publicly or bind to 0.0.0.0.
- Access Control: Session isolation relies on random UUID4 token unguessability for local demos.
  Conversation IDs are NOT a substitute for authenticated ownership or multi-tenant access control.
- ID Validation: At the API boundary (ChatRequest), invalid ID formats are strictly rejected with
  HTTP 422; omitted or None IDs automatically generate a fresh UUID4.
- Evidence Hierarchy: Prior assistant messages in conversation context are conversational
  reference only; they must NEVER be treated as trusted Knowledge Base evidence or as
  empirical proof that a vulnerability exists.
"""

from collections import OrderedDict
from dataclasses import dataclass, field
import datetime
import logging
import re
import threading
import uuid
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)

DEFAULT_MAX_CONVERSATIONS = 1000
DEFAULT_MAX_TURNS_PER_CONVERSATION = 6  # 3 user messages + 3 assistant responses
DEFAULT_MAX_MESSAGE_CHARS = 2000
VALID_CONVERSATION_ID_REGEX = re.compile(r"^[a-zA-Z0-9_\-]{1,64}$")


@dataclass
class ChatTurn:
    """A single message turn in a conversation."""
    role: str  # "user" or "assistant"
    content: str
    timestamp: datetime.datetime = field(default_factory=lambda: datetime.datetime.now(datetime.timezone.utc))


class InMemoryConversationStore:
    """Thread-safe, bounded, LRU in-memory conversation storage."""

    def __init__(
        self,
        max_conversations: int = DEFAULT_MAX_CONVERSATIONS,
        max_turns_per_conversation: int = DEFAULT_MAX_TURNS_PER_CONVERSATION,
        max_message_chars: int = DEFAULT_MAX_MESSAGE_CHARS,
    ):
        """Initialize in-memory store with strict bounds.

        Args:
            max_conversations: Maximum active sessions retained before LRU eviction.
            max_turns_per_conversation: Maximum message turns retained per session.
            max_message_chars: Maximum characters allowed per individual message.
        """
        self.max_conversations = max_conversations
        self.max_turns = max_turns_per_conversation
        self.max_message_chars = max_message_chars
        self._lock = threading.Lock()
        # OrderedDict used as LRU cache: most recently accessed at the end
        self._store: OrderedDict[str, List[ChatTurn]] = OrderedDict()

    def count_conversations(self) -> int:
        """Return total active sessions in memory."""
        with self._lock:
            return len(self._store)

    def validate_or_create_conversation_id(self, client_id: Optional[str]) -> str:
        """Validate client-supplied conversation ID or generate a fresh UUID4.

        SECURITY GUARANTEE:
        If a client supplies an ID that does not exist or has been evicted, a new
        session is initialized under that ID (or a new UUID4 if invalid), preventing
        unauthorized access to any other session.
        """
        if client_id and isinstance(client_id, str):
            clean_id = client_id.strip()
            if VALID_CONVERSATION_ID_REGEX.match(clean_id):
                with self._lock:
                    if clean_id not in self._store:
                        self._ensure_capacity_unlocked()
                        self._store[clean_id] = []
                    else:
                        self._store.move_to_end(clean_id)
                return clean_id

        # Generate fresh UUID4 if client_id is None, invalid format, or whitespace
        new_id = str(uuid.uuid4())
        with self._lock:
            self._ensure_capacity_unlocked()
            self._store[new_id] = []
        return new_id

    def _ensure_capacity_unlocked(self) -> None:
        """Evict oldest conversation if capacity reached (caller must hold lock)."""
        while len(self._store) >= self.max_conversations:
            oldest_id, _ = self._store.popitem(last=False)
            logger.debug("LRU evicted oldest conversation: %s", oldest_id)

    def add_turn(self, conversation_id: str, role: str, content: str) -> None:
        """Add a validated message turn to a conversation session with FIFO turn bounding.

        Args:
            conversation_id: Session identifier.
            role: 'user' or 'assistant'.
            content: Message body. Truncated to max_message_chars if oversized.
        """
        if role not in ("user", "assistant"):
            raise ValueError(f"Invalid message role '{role}'. Expected 'user' or 'assistant'.")

        bounded_content = (content or "").strip()[: self.max_message_chars]
        turn = ChatTurn(role=role, content=bounded_content)

        with self._lock:
            if conversation_id not in self._store:
                self._ensure_capacity_unlocked()
                self._store[conversation_id] = []

            turns = self._store[conversation_id]
            turns.append(turn)

            # FIFO bound on turns per conversation
            while len(turns) > self.max_turns:
                turns.pop(0)

            self._store.move_to_end(conversation_id)

    def get_history(self, conversation_id: str) -> List[ChatTurn]:
        """Retrieve copy of message turns for a conversation (isolated by session ID)."""
        with self._lock:
            if conversation_id not in self._store:
                return []
            self._store.move_to_end(conversation_id)
            return list(self._store[conversation_id])

    def get_formatted_context(self, conversation_id: str) -> Optional[str]:
        """Format historical turns into bounded text for prompt injection.

        Returns None if history is empty.
        """
        history = self.get_history(conversation_id)
        if not history:
            return None

        formatted_lines = []
        for turn in history:
            role_label = "User" if turn.role == "user" else "Assistant"
            formatted_lines.append(f"{role_label}: {turn.content}")

        return "\n".join(formatted_lines)

    def clear(self) -> None:
        """Clear all active sessions (used primarily in tests)."""
        with self._lock:
            self._store.clear()


# Default singleton conversation store
conversation_store = InMemoryConversationStore()
