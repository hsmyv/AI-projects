from app.core.config import LLM_PROVIDER
from app.services.openai_service import ask_ai
from app.services.ollama_service import ask_ollama


def ask_llm(question: str, context: str):

    if LLM_PROVIDER == "openai":
        return ask_ai(question, context)

    if LLM_PROVIDER == "ollama":
        return ask_ollama(question, context)

    raise Exception(f"Unsupported provider: {LLM_PROVIDER}")