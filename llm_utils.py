import os

from dotenv import load_dotenv
from groq import Groq

load_dotenv()

client = Groq(
    api_key=os.getenv("GROQ_API_KEY")
)


def generate_answer(question, retrieved_chunks):

    context = "\n\n".join(
        [
            f"Source: {chunk['document']}, Page: {chunk['page']}\n"
            f"{chunk['text']}"
            for chunk in retrieved_chunks
        ]
    )

    prompt = f"""
You are an academic document question-answering assistant.

Answer the user's question using ONLY the information provided in the context below.

If the answer cannot be found in the context, say:
"Information not found in the uploaded documents."

Do not use outside knowledge.
Do not make up information.

Context:
{context}

Question:
{question}

Answer:
"""

    response = client.chat.completions.create(
        model="openai/gpt-oss-20b",
        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ],
        temperature=0.1,
        max_completion_tokens=500
    )

    return response.choices[0].message.content