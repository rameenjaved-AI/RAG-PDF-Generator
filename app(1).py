import os
import io

import faiss
import numpy as np
import streamlit as st
import fitz  # PyMuPDF
from groq import Groq
from sentence_transformers import SentenceTransformer


st.set_page_config(
    page_title="PDF RAG Assistant",
    page_icon="📚",
    layout="wide",
)

st.title("📚 PDF RAG Assistant")
st.caption("Upload a PDF, build a FAISS vector index, and ask questions using Groq.")

EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
GROQ_MODEL = "llama-3.3-70b-versatile"
CHUNK_SIZE = 900
CHUNK_OVERLAP = 150
TOP_K = 5


@st.cache_resource
def load_embedding_model():
    return SentenceTransformer(EMBEDDING_MODEL)


def get_groq_client():
    api_key = None

    # Streamlit Cloud: st.secrets
    try:
        api_key = st.secrets.get("GROQ_API_KEY")
    except Exception:
        pass

    # Local development: environment variable
    if not api_key:
        api_key = os.getenv("GROQ_API_KEY")

    if not api_key:
        return None

    return Groq(api_key=api_key)


def extract_pdf_text(pdf_bytes):
    """Extract text page-by-page and keep page numbers for citations."""
    pages = []

    with fitz.open(stream=pdf_bytes, filetype="pdf") as doc:
        for page_number, page in enumerate(doc, start=1):
            text = page.get_text("text").strip()
            if text:
                pages.append(
                    {
                        "page": page_number,
                        "text": text,
                    }
                )

    return pages


def create_chunks(pages, chunk_size=CHUNK_SIZE, overlap=CHUNK_OVERLAP):
    """
    Character-based chunking with page metadata.
    The embedding model handles tokenization internally.
    """
    chunks = []

    for page in pages:
        text = " ".join(page["text"].split())

        if not text:
            continue

        start = 0
        text_length = len(text)

        while start < text_length:
            end = min(start + chunk_size, text_length)
            chunk_text = text[start:end].strip()

            if chunk_text:
                chunks.append(
                    {
                        "text": chunk_text,
                        "page": page["page"],
                    }
                )

            if end >= text_length:
                break

            start = max(0, end - overlap)

    return chunks


def build_faiss_index(chunks, embedding_model):
    texts = [chunk["text"] for chunk in chunks]

    embeddings = embedding_model.encode(
        texts,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=False,
    ).astype("float32")

    dimension = embeddings.shape[1]

    # Inner product on normalized vectors = cosine similarity.
    index = faiss.IndexFlatIP(dimension)
    index.add(embeddings)

    return index


def retrieve_chunks(question, index, chunks, embedding_model, top_k=TOP_K):
    query_embedding = embedding_model.encode(
        [question],
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=False,
    ).astype("float32")

    k = min(top_k, len(chunks))
    scores, indices = index.search(query_embedding, k)

    results = []

    for score, idx in zip(scores[0], indices[0]):
        if idx == -1:
            continue

        results.append(
            {
                "text": chunks[idx]["text"],
                "page": chunks[idx]["page"],
                "score": float(score),
            }
        )

    return results


def generate_answer(question, retrieved_chunks, client):
    context_parts = []

    for i, item in enumerate(retrieved_chunks, start=1):
        context_parts.append(
            f"[Source {i} | Page {item['page']}]\n{item['text']}"
        )

    context = "\n\n".join(context_parts)

    system_prompt = """You are a helpful PDF question-answering assistant.

Answer the user's question using ONLY the supplied document context.
If the answer cannot be found in the context, clearly say:
"I couldn't find that information in the uploaded PDF."

Do not invent facts.
Keep the answer clear and reasonably concise.
When possible, mention the relevant page number(s).
"""

    user_prompt = f"""Document context:

{context}

Question:
{question}
"""

    response = client.chat.completions.create(
        model=GROQ_MODEL,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0.1,
    )

    return response.choices[0].message.content


@st.cache_data(show_spinner=False)
def process_pdf(pdf_bytes):
    embedding_model = load_embedding_model()

    pages = extract_pdf_text(pdf_bytes)

    if not pages:
        return None, None, None

    chunks = create_chunks(pages)

    if not chunks:
        return None, None, None

    index = build_faiss_index(chunks, embedding_model)

    return pages, chunks, index


with st.sidebar:
    st.header("Settings")
    top_k = st.slider(
        "Retrieved chunks",
        min_value=1,
        max_value=10,
        value=TOP_K,
    )

    st.info(
        "Embeddings: all-MiniLM-L6-v2\n\n"
        "Vector DB: FAISS\n\n"
        "LLM: Llama 3.3 70B via Groq"
    )

uploaded_file = st.file_uploader(
    "Upload a PDF document",
    type=["pdf"],
)

if uploaded_file:
    pdf_bytes = uploaded_file.getvalue()

    with st.spinner("Extracting PDF, creating chunks, and building FAISS index..."):
        try:
            pages, chunks, index = process_pdf(pdf_bytes)
        except Exception as exc:
            st.error(f"Could not process the PDF: {exc}")
            st.stop()

    if not pages or not chunks or index is None:
        st.error(
            "No extractable text was found. This version works with "
            "text-based PDFs. Scanned/image-only PDFs need OCR."
        )
        st.stop()

    st.success(
        f"PDF processed successfully: {len(pages)} text pages, "
        f"{len(chunks)} chunks indexed."
    )

    client = get_groq_client()

    if client is None:
        st.warning(
            "GROQ_API_KEY is not configured. Add it to your local environment "
            "or Streamlit Cloud Secrets before asking questions."
        )
    else:
        question = st.chat_input("Ask a question about your PDF...")

        if question:
            with st.chat_message("user"):
                st.write(question)

            with st.chat_message("assistant"):
                with st.spinner("Searching the document and generating an answer..."):
                    try:
                        results = retrieve_chunks(
                            question,
                            index,
                            chunks,
                            load_embedding_model(),
                            top_k=top_k,
                        )

                        answer = generate_answer(
                            question,
                            results,
                            client,
                        )

                        st.write(answer)

                        with st.expander("Retrieved sources"):
                            for i, result in enumerate(results, start=1):
                                st.markdown(
                                    f"**Source {i} | Page {result['page']} | "
                                    f"Similarity {result['score']:.3f}**"
                                )
                                st.write(result["text"])

                    except Exception as exc:
                        st.error(f"Error while answering: {exc}")
else:
    st.info("Upload a PDF above to start chatting with your document.")
