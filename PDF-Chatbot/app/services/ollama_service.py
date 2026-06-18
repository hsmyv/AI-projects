import requests

OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL_NAME = "llama3.2:1b"


def ask_ollama(question: str, context: str):
    prompt = f"""
Sən PDF sənədi üzrə köməkçi botsan.

Aşağıdakı CONTEXT PDF-dən tapılıb.
Suala yalnız CONTEXT əsasında cavab ver.

CONTEXT:
{context}

SUAL:
{question}

CAVAB:
"""

    response = requests.post(
        OLLAMA_URL,
        json={
            "model": MODEL_NAME,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": 0.2,
                "num_predict": 300
            }
        },
        timeout=180
    )

    print("OLLAMA STATUS:", response.status_code)
    print("OLLAMA RAW:", response.text[:500])

    response.raise_for_status()

    return response.json().get("response", "").strip()