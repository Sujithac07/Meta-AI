from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import AsyncGenerator, List

@dataclass
class ChatMessage:
    role: str   # "user" | "assistant" | "system"
    content: str

class BaseLLMProvider(ABC):
    """All LLM providers must implement this interface."""
    
    @property
    @abstractmethod
    def name(self) -> str:
        """Provider identifier for logging."""
        pass
    
    @abstractmethod
    def is_available(self) -> bool:
        """Check if provider is configured and reachable."""
        pass
    
    @abstractmethod
    async def stream_chat(
        self,
        prompt: str,
        history: List[ChatMessage]
    ) -> AsyncGenerator[str, None]:
        """Stream response tokens one-by-one."""
        pass
    
    async def chat(
        self, 
        prompt: str, 
        history: List[ChatMessage]
    ) -> str:
        """Non-streaming fallback — collects all tokens."""
        tokens = []
        async for token in self.stream_chat(prompt, history):
            tokens.append(token)
        return "".join(tokens)
