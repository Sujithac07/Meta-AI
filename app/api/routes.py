from fastapi import APIRouter, File, UploadFile, HTTPException
from fastapi.responses import StreamingResponse
from app.api.schemas import ChatRequest, ChatResponse, ImageRequest, ImageResponse, IngestResponse, SessionStats
from app.providers.router import LLMRouter
from app.core.memory import ConversationMemory
from app.core.rag_engine import RAGEngine
from app.services.image_service import ImageService
import shutil
from pathlib import Path

router = APIRouter()
llm_router = LLMRouter()
rag_engine = RAGEngine()
image_service = ImageService()

@router.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    memory = ConversationMemory(request.session_id)
    history = memory.get_history()
    
    # Retrieval-Augmented Generation
    context = ""
    if request.rag_enabled:
        context = rag_engine.query(request.message) or ""
    
    full_prompt = request.message
    if context:
        full_prompt = f"CONTEXT FROM YOUR DOCUMENTS:\n{context}\n\nUSER QUESTION: {request.message}\n\nAnswer based on the context if possible."

    # Non-streaming response for this endpoint
    response_content = ""
    async for token in llm_router.chat(full_prompt, history):
        response_content += token
    
    memory.add_message("user", request.message)
    memory.add_message("assistant", response_content)
    
    return ChatResponse(
        session_id=request.session_id,
        message=response_content,
        sources=context if request.rag_enabled else None
    )

@router.post("/chat/stream")
async def chat_stream(request: ChatRequest):
    memory = ConversationMemory(request.session_id)
    history = memory.get_history()
    
    context = ""
    if request.rag_enabled:
        context = rag_engine.query(request.message) or ""
    
    full_prompt = request.message
    if context:
        full_prompt = f"CONTEXT FROM YOUR DOCUMENTS:\n{context}\n\nUSER QUESTION: {request.message}"

    async def generate():
        full_response = ""
        async for token in llm_router.chat(full_prompt, history):
            full_response += token
            yield token
        
        # Save to memory after the stream is finished
        memory.add_message("user", request.message)
        memory.add_message("assistant", full_response)

    return StreamingResponse(generate(), media_type="text/event-stream")

@router.post("/documents/ingest", response_model=IngestResponse)
async def ingest_document(file: UploadFile = File(...)):
    upload_dir = Path("data/uploads")
    upload_dir.mkdir(parents=True, exist_ok=True)
    
    file_path = upload_dir / file.filename
    with file_path.open("wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    
    try:
        chunks = rag_engine.ingest_document(str(file_path))
        return IngestResponse(filename=file.filename, chunks_created=chunks)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/images/generate", response_model=ImageResponse)
async def generate_image(request: ImageRequest):
    url = await image_service.generate_image(request.prompt)
    if not url:
        raise HTTPException(status_code=500, detail="Image generation failed or provider not available.")
    
    # Optional: Log to memory that an image was generated
    memory = ConversationMemory(request.session_id)
    memory.add_message("assistant", f"Generated image for: {request.prompt}", metadata={"image_url": url})
    
    return ImageResponse(
        session_id=request.session_id,
        image_url=url,
        prompt=request.prompt
    )

@router.get("/sessions/{session_id}/stats", response_model=SessionStats)
async def get_session_stats(session_id: str):
    memory = ConversationMemory(session_id)
    stats = memory.get_summary_stats()
    return SessionStats(session_id=session_id, **stats)

@router.delete("/sessions/{session_id}")
async def clear_session(session_id: str):
    memory = ConversationMemory(session_id)
    memory.clear()
    return {"status": "cleared", "session_id": session_id}
