import streamlit as st
import numpy as np
from sentence_transformers import SentenceTransformer


@st.cache_resource
def load_embedding_model():

    return SentenceTransformer(
        "all-MiniLM-L6-v2"
    )


def create_embeddings(
    chunks,
    progress_callback=None
):

    model = load_embedding_model()

    texts = [
        chunk["text"]
        for chunk in chunks
    ]

    batch_size = 32

    all_embeddings = []

    total_chunks = len(texts)

    for start in range(
        0,
        total_chunks,
        batch_size
    ):

        end = min(
            start + batch_size,
            total_chunks
        )

        batch_texts = texts[start:end]

        batch_embeddings = model.encode(
            batch_texts,
            normalize_embeddings=True,
            show_progress_bar=False
        )

        all_embeddings.append(
            batch_embeddings
        )

        if progress_callback:

            progress = end / total_chunks

            progress_callback(
                progress
            )

    return np.vstack(
        all_embeddings
    )


def create_query_embedding(question):

    model = load_embedding_model()

    embedding = model.encode(
        question,
        normalize_embeddings=True
    )

    return embedding