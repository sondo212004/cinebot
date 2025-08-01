from fastapi import FastAPI, Request
from pydantic import BaseModel
from fastapi.responses import StreamingResponse, JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from fastapi import HTTPException
from typing import AsyncGenerator
import sys
import os
import json
import uuid

# Thêm đường dẫn thư mục src vào Python path
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, ".."))
src_dir = os.path.join(project_root, "src")
sys.path.append(src_dir)

from chatbot_engine import ChatbotEngine


app = FastAPI()
engine = ChatbotEngine()

# Cấu hình CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # hoặc ["http://localhost:3000"] nếu bạn dùng frontend cụ thể
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class ChatRequest(BaseModel):
    message: str
    session_id: str = "default_session"

app.mount("/ui", StaticFiles(directory="ui"), name="ui")

@app.get("/chatbot-ui")
def get_chatbot_ui():
    return FileResponse("ui/chatbot_ui.html")

@app.get("/")
def root():
    return {"message": "CineBot API đang hoạt động 🎬"}

@app.post("/chat")
def chat(request: ChatRequest):
    response = engine.get_response(request.message, request.session_id)
    return {"response": response}

@app.post("/chat/stream")
async def stream_chat(request: ChatRequest):
    """
    Endpoint để stream phản hồi từ chatbot.
    """
    session_id = request.session_id if request.session_id else str(uuid.uuid4())

    if not hasattr(engine, 'app'):
        raise HTTPException(status_code=500, detail="ChatbotEngine chưa được khởi tạo. Vui lòng kiểm tra log.")

    async def generate_response_chunks() -> AsyncGenerator[str, None]:
        try:
            async for chunk_content in engine.stream_response(request.message, session_id):
                # Gói nội dung vào một đối tượng JSON và gửi đi
                yield f"data: {json.dumps({'content': chunk_content})}\n\n"
        except Exception as e:
            # Xử lý lỗi và gửi thông báo lỗi JSON
            yield f"data: {json.dumps({'error': str(e)})}\n\n"
        finally:
            yield "data: [DONE]\n\n" # Báo hiệu kết thúc stream

    return StreamingResponse(generate_response_chunks(), media_type="text/event-stream")

@app.get("/history/{session_id}")
def get_history(session_id: str):
    messages = engine.get_conversation_history(session_id)
    return JSONResponse([
        {"role":type(msg).__name__.replace("Message", "").lower(), "content": msg.content}
        for msg in messages
    ])

@app.delete("/history/{session_id}")
def delete_history(session_id: str):
    return {"message": engine.clear_conversation(session_id) or "Lịch sử trò chuyện đã được xóa."}

