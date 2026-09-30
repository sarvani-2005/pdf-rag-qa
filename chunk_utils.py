from langchain_text_splitters import RecursiveCharacterTextSplitter


def create_chunks(pages):

    splitter = RecursiveCharacterTextSplitter(
        separators=[
            "\n\n",
            "\n",
            ". ",
            "? ",
            "! ",
            " ",
            ""
        ],
        chunk_size=1500,
        chunk_overlap=300
    )

    chunks = []

    for page in pages:

        page_chunks = splitter.split_text(
            page["text"]
        )

        for chunk_text in page_chunks:

            chunks.append({
                "document": page["document"],
                "page": page["page"],
                "text": chunk_text.strip()
            })

    return chunks