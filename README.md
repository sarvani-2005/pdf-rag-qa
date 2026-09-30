# 📄 PDF RAG Question Answering System

A Retrieval-Augmented Generation (RAG) application that allows users to upload PDF documents and ask questions about their content.

## ✨ Features

- 📄 Upload one or multiple PDF documents
- 🔎 Semantic search using FAISS
- 🧠 Text embeddings using `all-MiniLM-L6-v2`
- 🤖 Answer generation using GPT-OSS-20B via Groq
- 📚 Displays source document and page number
- 🛡️ Handles questions when relevant information is not found

## 🔄 How It Works

```text
PDF Upload
    ↓
Text Extraction
    ↓
Chunking
    ↓
Embeddings
    ↓
FAISS Retrieval
    ↓
Relevant Context
    ↓
GPT-OSS-20B
    ↓
Answer + Sources
```

## 🛠️ Tech Stack

**Python · Streamlit · pypdf · LangChain · Sentence Transformers · FAISS · Groq · GPT-OSS-20B**


## ⚙️ Setup

```bash
git clone https://github.com/YOUR_USERNAME/pdf-rag-qa.git
cd pdf-rag-qa

python -m venv venv
venv\Scripts\activate

pip install -r requirements.txt
```

Create a `.env` file:

```text
GROQ_API_KEY=your_groq_api_key
```

Run the application:

```bash
python -m streamlit run app.py
```

## ⚠️ Limitations

- Scanned/image-only PDFs are not currently supported.
- Complex tables and multi-column layouts may not extract perfectly.
- Retrieval quality depends on the embedding model and similarity threshold.
