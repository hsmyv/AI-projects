from fastapi import APIRouter, Request, UploadFile, File, Form
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates

from app.services.chunk_service import split_text_into_chunks
from app.services.llm_service import ask_llm
from app.services.pdf_service import extract_text_from_pdf
from app.services.embedding_service import create_embedding
from app.services.vector_service import (
    save_chunks_to_vector_db,
    search_similar_chunks,
    delete_document_chunks
)

router = APIRouter()
templates = Jinja2Templates(directory="app/templates")


@router.get("/", response_class=HTMLResponse)
def home(request: Request):
    return templates.TemplateResponse(
        request,
        "index.html",
        {
            "answer": None,
            "chunks": None
        }
    )


@router.post("/", response_class=HTMLResponse)
async def chat_with_pdf(
    request: Request,
    pdf: UploadFile = File(None),
    question: str = Form(None)
):
    answer = None
    chunks = None

    if pdf:
        filename = pdf.filename

        text = await extract_text_from_pdf(pdf)
        pdf_chunks = split_text_into_chunks(text)

        embeddings = []
        for chunk in pdf_chunks:
            embeddings.append(create_embedding(chunk))

        delete_document_chunks(filename)

        save_chunks_to_vector_db(
            chunks=pdf_chunks,
            embeddings=embeddings,
            filename=filename
        )

    if question:
        question_embedding = create_embedding(question)

        results = search_similar_chunks(
            question_embedding=question_embedding,
            top_k=3
        )

        chunks = results["documents"]

        context = "\n\n".join(chunks)

        answer = ask_llm(question, context)


    return templates.TemplateResponse(
        request,
        "index.html",
        {
            "answer": answer,
            "chunks": chunks
        }
    )