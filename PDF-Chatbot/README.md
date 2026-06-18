# PDF Chatbot RAG

A fully local Retrieval-Augmented Generation (RAG) chatbot that allows users to upload PDF documents and ask questions about their content.

The system extracts text from PDFs, splits it into chunks, generates embeddings, stores them in a vector database, retrieves the most relevant context, and generates answers using a configurable LLM provider.

## Features

* Upload PDF documents
* Extract and process PDF text
* Intelligent text chunking with overlap
* Embedding generation using Sentence Transformers
* Vector search with ChromaDB
* Retrieval-Augmented Generation (RAG)
* Configurable LLM provider support

  * Ollama (default)
  * OpenAI
* FastAPI REST API
* Local-first architecture

## Architecture

PDF Upload

↓

Text Extraction (PyMuPDF)

↓

Chunking

↓

Embeddings (Sentence Transformers)

↓

ChromaDB

↓

Similarity Search

↓

LLM (Ollama / OpenAI)

↓

Answer

## Tech Stack

### Backend

* FastAPI
* Python

### AI & RAG

* Sentence Transformers
* ChromaDB
* Ollama
* OpenAI (optional)

### PDF Processing

* PyMuPDF

## Project Structure

```text
app/
├── core/
├── routes/
├── services/
│   ├── pdf_service.py
│   ├── chunk_service.py
│   ├── embedding_service.py
│   ├── vector_service.py
│   ├── ollama_service.py
│   ├── openai_service.py
│   └── llm_service.py
│
├── main.py

uploads/
chroma_db/
```

## LLM Providers

The application supports multiple LLM providers through a unified interface.

Example configuration:

```env
LLM_PROVIDER=ollama
```

Available providers:

* ollama
* openai

## API Endpoints

### Upload PDF

```http
POST /upload
```

Uploads and indexes a PDF document.

### Chat

```http
POST /chat
```

Request:

```json
{
  "question": "What is this document about?"
}
```

Response:

```json
{
  "question": "What is this document about?",
  "answer": "...",
  "sources": [...]
}
```

## Installation

```bash
git clone <repository-url>

cd pdf-chatbot

python -m venv venv

venv\Scripts\activate

pip install -r requirements.txt
```

Run the application:

```bash
uvicorn app.main:app --reload
```

API documentation:

```text
http://127.0.0.1:8000/docs
```

## Future Improvements

* Source citations with page numbers
* Multi-document conversations
* Chat history support
* User authentication
* Document management
* Streaming responses
* Hybrid search (keyword + vector)
* Support for additional LLM providers

## License

MIT License
