from embedding_utils import create_query_embedding
from vector_store import search_vector_store
from llm_utils import generate_answer


def retrieve_relevant_chunks(
    question,
    index,
    chunks,
    top_k=5
):

    question_embedding = create_query_embedding(
        question
    )

    results = search_vector_store(
        index,
        question_embedding,
        chunks,
        top_k=top_k,
        min_score=0.50
    )

    return results


def answer_question(
    question,
    index,
    chunks,
    top_k=5
):

    retrieved_chunks = retrieve_relevant_chunks(
        question,
        index,
        chunks,
        top_k
    )

    # No sufficiently relevant chunks
    if not retrieved_chunks:

        return (
            "Information not found in the uploaded documents.",
            []
        )

    answer = generate_answer(
        question,
        retrieved_chunks
    )

    return answer, retrieved_chunks