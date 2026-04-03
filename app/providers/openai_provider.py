import os
from typing import AsyncGenerator, List
from app.providers.base import BaseLLMProvider, ChatMessage
from app.utils.logger import get_logger
from app.config import get_settings

logger = get_logger(__name__)
settings = get_settings()

class OpenAIProvider(BaseLLMProvider):
    
    @property
    def name(self) -> str:
        return "OpenAI"
    
    def is_available(self) -> bool:
        return bool(settings.OPENAI_API_KEY)
    
    async def stream_chat(
        self,
        prompt: str,
        history: List[ChatMessage]
    ) -> AsyncGenerator[str, None]:
        from openai import AsyncOpenAI
        
        client = AsyncOpenAI(api_key=settings.OPENAI_API_KEY)
        
        messages = [{"role": m.role, "content": m.content} for m in history]
        messages.append({"role": "user", "content": prompt})
        
        response = await client.chat.completions.create(
            model=os.getenv("OPENAI_MODEL", "gpt-4o"),
            messages=messages,
            stream=True
        )
        
        async for chunk in response:
            if chunk.choices[0].delta.content:
                yield chunk.choices[0].delta.content
