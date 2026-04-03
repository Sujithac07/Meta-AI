"""
Multi-Modal Service Layer for image generation and vision tasks.
"""
import asyncio
from typing import Optional
from app.utils.logger import get_logger
from app.utils.retry import retry_with_backoff

logger = get_logger(__name__)

class ImageService:
    """Handle image generation and analysis using Meta AI or Fallbacks."""
    
    def __init__(self):
        try:
            from meta_ai_api import MetaAI
            self.agent = MetaAI()
        except ImportError:
            self.agent = None
            logger.warning("meta-ai-api not installed. Image generation limited.")

    @retry_with_backoff(max_attempts=2)
    async def generate_image(self, prompt: str) -> Optional[str]:
        """Generate an image from text. Returns URL or base64."""
        if not self.agent:
            return None
            
        logger.info(f"Generating image for prompt: {prompt[:50]}...")
        
        loop = asyncio.get_event_loop()
        try:
            # The unofficial API might return a URL or a dictionary
            # Meta AI image generation prompt typically starts with or includes specific triggers
            response = await loop.run_in_executor(
                None,
                lambda: self.agent.image(prompt=prompt)
            )
            
            # Extract URL if present
            if isinstance(response, dict):
                return response.get("url") or response.get("image_url")
            return response
            
        except Exception as e:
            logger.error(f"Image generation failed: {e}")
            return None

    async def analyze_image(self, image_path: str, question: str) -> Optional[str]:
        """Analyze an image using vision capabilities."""
        # Note: Unofficial Meta AI API might have limited vision support
        # This could fall back to OpenAI GPT-4o vision if available
        from app.config import get_settings
        settings = get_settings()
        
        if settings.OPENAI_API_KEY:
            logger.info("Using OpenAI for image vision analysis fallback.")
            from openai import AsyncOpenAI
            import base64
            
            client = AsyncOpenAI(api_key=settings.OPENAI_API_KEY)
            
            with open(image_path, "rb") as image_file:
                base64_image = base64.b64encode(image_file.read()).decode('utf-8')
            
            response = await client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": question},
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/jpeg;base64,{base64_image}"
                                }
                            }
                        ]
                    }
                ]
            )
            return response.choices[0].message.content
            
        return "Vision analysis not implemented for current configuration."
