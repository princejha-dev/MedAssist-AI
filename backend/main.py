from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from backend.model import assistant


class ChatRequest(BaseModel):
    question: str


app = FastAPI(title="Medical RAG Chatbot API")

# allow requests from local frontend during development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
async def health() -> dict:
    return {"status": "ok"}


@app.post("/chat")
async def chat(req: ChatRequest) -> dict:
    """Receive a user question and return the model answer."""

    if not req.question.strip():
        raise HTTPException(status_code=400, detail="Empty question provided.")

    try:
        answer = assistant.ask(req.question)
        return {"answer": answer}
    except Exception as exc:
        # log the error in a real application
        raise HTTPException(status_code=500, detail=str(exc))
