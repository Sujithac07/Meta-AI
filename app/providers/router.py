"""
LLM Provider Router — Supports automatic fallback between providers.
If Meta AI fails, automatically routes to Groq (fast) → OpenAI (reliable).
"""
from typing import AsyncGenerator, List
from app.providers.base import BaseLLMProvider, ChatMessage
from app.providers.meta_provider import MetaAIProvider
from app.providers.groq_provider import GroqProvider
from app.providers.openai_provider import OpenAIProvider
from app.utils.logger import get_logger

logger = get_logger(__name__)

class LLMRouter:
    """Smart router with automatic provider failover."""
    
    def __init__(self):
        self._providers: List[BaseLLMProvider] = [
            MetaAIProvider(),    # Primary (free, uses Meta's web interface)
            GroqProvider(),      # Fast inference fallback (LLaMA 3, Mixtral)
            OpenAIProvider(),    # Reliable fallback (GPT-4o)
        ]
    
    async def chat(
        self, 
        prompt: str, 
        history: List[ChatMessage],
        stream: bool = True
    ) -> AsyncGenerator[str, None]:
        """Route request to best available provider with fallback."""
        
        last_error = None
        for provider in self._providers:
            if not provider.is_available():
                logger.debug(f"Provider {provider.name} is not available/configured. Skipping.")
                continue
            try:
                logger.info(f"Routing to provider: {provider.name}")
                async for token in provider.stream_chat(prompt, history):
                    yield token
                return  # Success — stop trying other providers
            except Exception as e:
                last_error = e
                logger.warning(
                    f"Provider {provider.name} failed: {e}. "
                    f"Trying next provider..."
                )
                continue
        
        raise RuntimeError(
            f"All providers exhausted. Last error: {last_error}"
        )
