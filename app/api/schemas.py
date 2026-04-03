from pydantic import BaseModel
from typing import Optional

class ChatMessageSchema(BaseModel):
    role: str
    content: str

class ChatRequest(BaseModel):
    session_id: str
    message: str
    rag_enabled: bool = False

class ChatResponse(BaseModel):
    session_id: str
    message: str
    sources: Optional[str] = None

class ImageRequest(BaseModel):
    session_id: str
    prompt: str

class ImageResponse(BaseModel):
    session_id: str
    image_url: str
    prompt: str

class IngestResponse(BaseModel):
    filename: str
    chunks_created: int
    status: str = "success"

class SessionStats(BaseModel):
    session_id: str
    total_messages: int
    first_message: Optional[str]
    last_message: Optional[str]
