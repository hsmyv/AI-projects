from fastapi import APIRouter, UploadFile, File, HTTPException
from app.services.pdf_service import extract_text_from_pdf
from app.services.chunk_service import split_text_into_chunks
from app.services.embedding_service import create_embeddings
from app.services.vector_service import (
    save_chunks_to_vector_db,
    delete_document_chunks
)
import os

router = APIRouter()

UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)


@router.post("/upload")
async def upload_pdf(file: UploadFile = File(...)):

    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(
            status_code=400,
            detail="Only PDF files are allowed"
        )

    file_path = os.path.join(UPLOAD_FOLDER, file.filename)

    with open(file_path, "wb") as buffer:
        buffer.write(await file.read())

    deleted_count = delete_document_chunks(file.filename)

    text = extract_text_from_pdf(file_path)

    if not text.strip():
        raise HTTPException(
            status_code=400,
            detail="No extractable text found in PDF"
        )

    chunks = split_text_into_chunks(text)

    if not chunks:
        raise HTTPException(
            status_code=400,
            detail="No valid chunks generated from PDF"
        )

    embeddings = create_embeddings(chunks)

    saved_count = save_chunks_to_vector_db(
        chunks=chunks,
        embeddings=embeddings,
        filename=file.filename
    )

    return {
        "message": "PDF processed and saved to vector database",
        "deleted_chunks": deleted_count,
        "filename": file.filename,
        "characters": len(text),
        "chunks_count": len(chunks),
        "saved_chunks": saved_count
    }