from fastapi import APIRouter
from pydantic import BaseModel
from app.services.embedding_service import create_embedding
from app.services.llm_service import ask_llm
from app.services.vector_service import search_similar_chunks

router = APIRouter()


class ChatRequest(BaseModel):
    question: str


@router.post("/chat")
def chat(request: ChatRequest):
    question_embedding = create_embedding(request.question)

    results = search_similar_chunks(question_embedding)

    context = "\n\n".join(results["documents"])

    answer = ask_llm(
        question=request.question,
        context=context
    )

    return {
        "question": request.question,
        "answer": answer,
        "matched_chunks": results["documents"],
        "sources": results["metadatas"]
    }