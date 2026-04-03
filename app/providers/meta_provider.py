import asyncio
import importlib.util
from typing import AsyncGenerator, List
from app.providers.base import BaseLLMProvider, ChatMessage
from app.utils.retry import retry_with_backoff
from app.utils.logger import get_logger
from app.config import get_settings

logger = get_logger(__name__)
settings = get_settings()

class MetaAIProvider(BaseLLMProvider):
    
    @property
    def name(self) -> str:
        return "MetaAI"
    
    def is_available(self) -> bool:
        # In a real scenario, we'd also verify cookie/auth setup.
        return importlib.util.find_spec("meta_ai_api") is not None
    
    @retry_with_backoff(max_attempts=3, base_delay=2.0)
    async def stream_chat(
        self,
        prompt: str,
        history: List[ChatMessage]
    ) -> AsyncGenerator[str, None]:
        try:
            from meta_ai_api import MetaAI
        except ImportError:
            raise RuntimeError("meta-ai-api not installed")
        
        # Build context-aware prompt with history
        context = "\n".join([
            f"{m.role.upper()}: {m.content}" 
            for m in history[-10:]  # Last 10 turns only
        ])
        full_prompt = f"{context}\nUSER: {prompt}" if context else prompt
        
        # Run blocking API call in thread pool (non-blocking)
        loop = asyncio.get_event_loop()
        
        # Initialize MetaAI with potential cookie from settings
        ai = MetaAI() # You might need to pass cookies logic here if needed
        
        response = await loop.run_in_executor(
            None,
            lambda: ai.prompt(message=full_prompt)
        )
        
        # Simulate streaming by yielding chunks if the API doesn't support it natively
        message = response.get("message", "")
        if not message and "error" in response:
            raise RuntimeError(f"MetaAI Error: {response['error']}")
            
        chunk_size = 20
        for i in range(0, len(message), chunk_size):
            yield message[i:i + chunk_size]
            await asyncio.sleep(0.01)  # Yield control
