import faiss
import numpy as np


def create_vector_store(embeddings):

    embeddings = np.asarray(
        embeddings
    ).astype("float32")

    dimension = embeddings.shape[1]

    index = faiss.IndexFlatIP(dimension)

    index.add(embeddings)

    return index


def search_vector_store(
    index,
    query_embedding,
    chunks,
    top_k=5,
    min_score=0.50
):

    query_embedding = np.asarray(
        [query_embedding]
    ).astype("float32")

    scores, indices = index.search(
        query_embedding,
        top_k
    )

    results = []

    for score, index_position in zip(
        scores[0],
        indices[0]
    ):

        if index_position == -1:
            continue

        # Keep only sufficiently relevant chunks
        if score < min_score:
            continue

        results.append({
            "score": float(score),
            "document": chunks[index_position]["document"],
            "page": chunks[index_position]["page"],
            "text": chunks[index_position]["text"]
        })

    return results