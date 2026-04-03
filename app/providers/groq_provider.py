"""
Groq Provider — Fast inference fallback for META-AI.
Groq provides extremely fast LLaMA and Mixtral models.
"""
import os
from typing import AsyncGenerator, List
from app.providers.base import BaseLLMProvider, ChatMessage
from app.utils.logger import get_logger
from app.config import get_settings

logger = get_logger(__name__)
settings = get_settings()

class GroqProvider(BaseLLMProvider):
    """
    Groq provider for fast inference.
    Falls back to this when Meta AI or OpenAI are unavailable/slow.
    """
    
    @property
    def name(self) -> str:
        return "Groq"
    
    def is_available(self) -> bool:
        """Check if Groq is configured with API key."""
        return bool(settings.GROQ_API_KEY)
    
    async def stream_chat(
        self,
        prompt: str,
        history: List[ChatMessage]
    ) -> AsyncGenerator[str, None]:
        """
        Stream chat using Groq's async API.
        Uses LLaMA 3 or Mixtral for fast inference.
        """
        from groq import AsyncGroq
        
        client = AsyncGroq(api_key=settings.GROQ_API_KEY)
        
        # Build conversation history
        messages = [{"role": m.role, "content": m.content} for m in history]
        messages.append({"role": "user", "content": prompt})
        
        # Use Groq's fast inference models
        # llama3-70b-8192 is fast and capable
        # mixtral-8x7b-32768 is also available for longer context
        model = os.getenv("GROQ_MODEL", "llama-3.1-70b-versatile")
        
        logger.info(f"Groq provider: Using model {model}")
        
        response = await client.chat.completions.create(
            model=model,
            messages=messages,
            stream=True,
            temperature=float(os.getenv("LLM_TEMPERATURE", "0.7")),
            max_tokens=int(os.getenv("LLM_MAX_TOKENS", "4096")),
            top_p=float(os.getenv("LLM_TOP_P", "0.9")),
        )
        
        async for chunk in response:
            if chunk.choices[0].delta.content:
                yield chunk.choices[0].delta.content
