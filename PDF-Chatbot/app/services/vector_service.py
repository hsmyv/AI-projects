import chromadb

chroma_client = chromadb.PersistentClient(path="chroma_db")

collection = chroma_client.get_or_create_collection(name="pdf_chunks")


def save_chunks_to_vector_db(chunks: list[str], embeddings: list[list[float]], filename: str):
    ids = []
    metadatas = []

    for index, chunk in enumerate(chunks):
        ids.append(f"{filename}_{index}")
        metadatas.append({
            "filename": filename,
            "chunk_index": index
        })

    collection.add(
        ids=ids,
        documents=chunks,
        embeddings=embeddings,
        metadatas=metadatas
    )

    return len(chunks)



def search_similar_chunks(question_embedding: list[float], top_k: int = 3):
    results = collection.query(
        query_embeddings=[question_embedding],
        n_results=top_k
    )

    return {
        "documents": results["documents"][0],
        "metadatas": results["metadatas"][0],
        "distances": results["distances"][0]
    }


def delete_document_chunks(filename: str):

    results = collection.get(
        where={"filename": filename}
    )

    ids = results["ids"]

    if ids:
        collection.delete(ids=ids)

    return len(ids)