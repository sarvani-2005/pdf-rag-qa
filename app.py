import streamlit as st

from pdf_utils import extract_pdf_text
from chunk_utils import create_chunks
from embedding_utils import create_embeddings
from vector_store import create_vector_store
from rag_pipeline import answer_question


# --------------------------------------------------
# Page configuration
# --------------------------------------------------

st.set_page_config(
    page_title="Academic RAG QA",
    page_icon="📚",
    layout="wide"
)

st.title("📚 Academic Document QA")

st.write(
    "Upload academic PDF documents and ask questions about their content."
)


# --------------------------------------------------
# PDF Upload
# --------------------------------------------------

uploaded_files = st.file_uploader(
    "Upload PDF documents",
    type=["pdf"],
    accept_multiple_files=True
)


# --------------------------------------------------
# Instructions shown before uploading
# --------------------------------------------------

if not uploaded_files:

    st.info(
        """
        👆 Upload one or more PDF documents to begin.

        **How it works:**

        1. 📄 Extract text from your PDFs
        2. ✂️ Split the text into smaller chunks
        3. 🧠 Convert chunks into embeddings
        4. 🔎 Store them in FAISS for similarity search
        5. 💬 Ask questions about your documents
        """
    )


# --------------------------------------------------
# Process uploaded documents
# --------------------------------------------------

if uploaded_files:

    st.write(
        f"📄 **{len(uploaded_files)} document(s) selected**"
    )

    if st.button(
        "🚀 Process Documents",
        type="primary"
    ):

        with st.status(
            "🔄 Processing your documents...",
            expanded=True
        ) as status:

            # ------------------------------------------
            # Step 1: Extract text from PDFs
            # ------------------------------------------

            st.write(
                "📄 Extracting text from PDF documents..."
            )

            all_pages = []

            extraction_progress = st.progress(
                0,
                text="Extracting documents..."
            )

            total_files = len(uploaded_files)

            for file_number, uploaded_file in enumerate(
                uploaded_files,
                start=1
            ):

                pages = extract_pdf_text(uploaded_file)

                all_pages.extend(pages)

                progress = file_number / total_files

                extraction_progress.progress(
                    progress,
                    text=(
                        f"Extracted "
                        f"{file_number}/{total_files} document(s)"
                    )
                )

            st.write(
                f"✅ Extracted text from "
                f"**{len(all_pages)} pages**."
            )


            # ------------------------------------------
            # Step 2: Create chunks
            # ------------------------------------------

            st.write(
                "✂️ Splitting document text into chunks..."
            )

            chunks = create_chunks(all_pages)

            st.write(
                f"✅ Created **{len(chunks)} chunks**."
            )


            # ------------------------------------------
            # Step 3: Generate embeddings
            # ------------------------------------------

            st.write(
                "🧠 Generating embeddings..."
            )

            embedding_progress = st.progress(
                0,
                text="Generating embeddings..."
            )


            def update_embedding_progress(progress):

                percentage = int(progress * 100)

                embedding_progress.progress(
                    progress,
                    text=(
                        f"Generating embeddings... "
                        f"{percentage}%"
                    )
                )


            embeddings = create_embeddings(
                chunks,
                progress_callback=update_embedding_progress
            )

            st.write(
                "✅ Embeddings generated."
            )


            # ------------------------------------------
            # Step 4: Create FAISS index
            # ------------------------------------------

            st.write(
                "🔎 Building FAISS vector index..."
            )

            index = create_vector_store(
                embeddings
            )

            st.write(
                "✅ FAISS index created."
            )


            # ------------------------------------------
            # Store everything in Streamlit session
            # ------------------------------------------

            st.session_state["chunks"] = chunks

            st.session_state["index"] = index

            st.session_state["documents_processed"] = (
                len(uploaded_files)
            )


            status.update(
                label="✅ Documents processed successfully!",
                state="complete",
                expanded=False
            )


        st.success(
            f"Ready! You can now ask questions about "
            f"your {len(uploaded_files)} document(s)."
        )


# --------------------------------------------------
# Question Answering
# --------------------------------------------------

if "index" in st.session_state:

    st.divider()

    st.subheader("💬 Ask a Question")

    question = st.text_input(
        "Ask something about your uploaded documents:"
    )


    if question:

        # ------------------------------------------
        # Retrieve chunks + generate answer
        # ------------------------------------------

        with st.spinner(
            "🔎 Searching documents and generating answer..."
        ):

            answer, results = answer_question(
                question,
                st.session_state["index"],
                st.session_state["chunks"],
                top_k=5
            )


        # ------------------------------------------
        # Display generated answer
        # ------------------------------------------

        st.subheader("🤖 Answer")

        st.write(answer)


        # ------------------------------------------
        # Display retrieved sources
        # ------------------------------------------

        st.subheader("📚 Sources")

        if results:

            for number, result in enumerate(
                results,
                start=1
            ):

                st.markdown(
                    f"### Source {number}"
                )

                st.write(
                    f"**Document:** "
                    f"{result['document']}"
                )

                st.write(
                    f"**Page:** "
                    f"{result['page']}"
                )

                st.write(
                    f"**Similarity:** "
                    f"{result['score']:.3f}"
                )

                st.write(
                    result["text"]
                )

                st.divider()

        else:

            st.warning(
                "No relevant information was found "
                "in the uploaded documents."
            )