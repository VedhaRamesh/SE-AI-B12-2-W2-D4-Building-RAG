import os
import re
import tempfile
from typing import List, Dict, Any, Optional
from urllib.parse import urlparse

import streamlit as st
from dotenv import load_dotenv

load_dotenv()
os.environ.setdefault("USER_AGENT", "MultiSourceRAGAssistant/1.0")

from langchain_community.document_loaders import PyPDFLoader, WebBaseLoader
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI, OpenAIEmbeddings


def get_openai_api_key() -> Optional[str]:
    """Read the API key from the environment variable expected by the assignment."""
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key or not api_key.strip():
        return None
    return api_key.strip()


def get_safe_file_name(source: Optional[str]) -> str:
    """Return a readable filename even when source metadata is missing."""
    if not source:
        return "Unknown source"

    if isinstance(source, str) and re.match(r"^https?://", source, flags=re.IGNORECASE):
        parsed = urlparse(source)
        host = parsed.netloc or "website"
        path = parsed.path.rstrip("/")
        if path and path != "/":
            return f"{host}{path}"
        return host

    return os.path.basename(source)


def normalize_page_numbers(documents: List[Document]) -> None:
    """Convert page metadata to human-readable page numbers starting at 1.

    PyPDFLoader often stores page numbers as 0-based values internally, so we detect
    that pattern and shift them to the standard 1-based page numbering used in UI.
    """
    page_values: List[int] = []
    for doc in documents:
        page_value = doc.metadata.get("page")
        try:
            page_values.append(int(page_value))
        except (TypeError, ValueError):
            continue

    offset = 1 if page_values and min(page_values) == 0 else 0

    for doc in documents:
        page_value = doc.metadata.get("page")
        try:
            page_number = int(page_value)
        except (TypeError, ValueError):
            page_number = 1
        doc.metadata["page"] = max(1, page_number + offset)
        doc.metadata["filename"] = get_safe_file_name(doc.metadata.get("source"))
        doc.metadata["source"] = doc.metadata.get("filename", "Unknown source")


def read_pdf_documents(uploaded_file) -> List[Document]:
    """Load a single PDF using PyPDFLoader and keep metadata for source + pages."""
    temp_file = None
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_handle:
            temp_handle.write(uploaded_file.getvalue())
            temp_file = temp_handle.name

        loader = PyPDFLoader(temp_file)
        docs = loader.load()

        if not docs:
            st.warning(f"{uploaded_file.name}: No readable text was found in the PDF.")
            return []

        for doc in docs:
            doc.metadata["source"] = uploaded_file.name
            doc.metadata["filename"] = uploaded_file.name

        normalize_page_numbers(docs)
        return docs
    except Exception:
        st.warning(f"{uploaded_file.name}: The PDF could not be read. It may be empty, unreadable, or password-protected.")
        return []
    finally:
        if temp_file and os.path.exists(temp_file):
            os.remove(temp_file)


def read_website_documents(url: str) -> List[Document]:
    """Load website content from a URL and store metadata for citation tracking."""
    cleaned_url = url.strip()
    if not cleaned_url:
        return []

    if not re.match(r"^https?://", cleaned_url, flags=re.IGNORECASE):
        cleaned_url = f"https://{cleaned_url}"

    try:
        loader = WebBaseLoader(
            [cleaned_url],
            requests_kwargs={
                "headers": {
                    "User-Agent": (
                        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                        "AppleWebKit/537.36 (KHTML, like Gecko) "
                        "Chrome/131.0.0.0 Safari/537.36"
                    ),
                    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
                },
                "timeout": 30,
            },
        )
        docs = loader.load()
    except Exception as exc:
        st.warning(
            f"{cleaned_url}: The page could not be fetched ({type(exc).__name__}: {exc}). "
            "The site may be temporarily unavailable or blocking automated requests."
        )
        return []

    if not docs:
        st.warning(f"{cleaned_url}: No readable text was found on the page.")
        return []

    for doc in docs:
        doc.metadata["source"] = cleaned_url
        doc.metadata["filename"] = get_safe_file_name(cleaned_url)
        doc.metadata["page"] = 1

    return docs


def build_vector_store(documents: List[Document], embeddings: OpenAIEmbeddings):
    """Split PDFs into chunks and create a single FAISS vector database."""
    if not documents:
        return None, 0

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=150,
        length_function=len,
        separators=["\n\n", "\n", " ", ""],
    )
    chunks = splitter.split_documents(documents)
    try:
        vector_store = FAISS.from_documents(chunks, embeddings)
    except Exception as exc:
        raise RuntimeError("Embedding or vector indexing failed. Please verify your OPENAI_API_KEY and try again.") from exc
    return vector_store, len(chunks)


def unique_citations(documents: List[Document]) -> List[Dict[str, Any]]:
    """Deduplicate filename and page citations from retrieved chunks."""
    seen = set()
    citations: List[Dict[str, Any]] = []
    for doc in documents:
        filename = get_safe_file_name(doc.metadata.get("source"))
        page_number = int(doc.metadata.get("page", 1) or 1)
        identifier = (filename, page_number)
        if identifier in seen:
            continue
        seen.add(identifier)
        citations.append({"filename": filename, "page": page_number})
    return citations


def split_sentences(text: str) -> List[str]:
    """Split text into sentence-like blocks for conflict detection."""
    sentences = re.split(r"(?<=[.!?])\s+|\n+", text)
    cleaned = []
    for sentence in sentences:
        cleaned_sentence = sentence.strip()
        if not cleaned_sentence:
            continue
        if re.fullmatch(r"page\s+\d+", cleaned_sentence, flags=re.IGNORECASE):
            continue
        if re.fullmatch(r"\d+", cleaned_sentence):
            continue
        cleaned.append(cleaned_sentence)
    return cleaned


def sentence_key(sentence: str) -> str:
    """Create a rough subject key from a sentence so similar claims can be compared."""
    match = re.search(r"(\d+(?:\.\d+)?)\s*(days?|months?|years?|weeks?|hours?|minutes?|percent|%)?", sentence, flags=re.IGNORECASE)
    if match:
        prefix = sentence[: match.start()]
        cleaned = re.sub(r"[^a-z0-9\s]", " ", prefix.lower())
        cleaned = re.sub(r"\s+", " ", cleaned).strip()
        if cleaned:
            words = cleaned.split()
            return " ".join(words[-8:])
    return re.sub(r"[^a-z0-9\s]", " ", sentence.lower())


def detect_conflicts(question: str, documents: List[Document]) -> List[Dict[str, Any]]:
    """Heuristically detect contradictory statements when multiple documents mention different values.

    This is intentionally simple and explainable for beginner use. It looks for the same
    subject in different sources with different numerical values, which is a common pattern
    in policy and handbook disagreements.
    """
    question_terms = set(term for term in re.findall(r"[A-Za-z]{4,}", question.lower()) if len(term) >= 4)
    candidates: List[Dict[str, Any]] = []

    for doc in documents:
        source = get_safe_file_name(doc.metadata.get("source"))
        page = int(doc.metadata.get("page", 1) or 1)
        for sentence in split_sentences(doc.page_content):
            sentence_lower = sentence.lower()
            has_number = bool(re.search(r"\d+", sentence))
            if not has_number:
                continue
            if question_terms and not any(term in sentence_lower for term in question_terms):
                continue
            number_match = re.search(r"(\d+(?:\.\d+)?)\s*(days?|months?|years?|weeks?|hours?|minutes?|percent|%)?", sentence, flags=re.IGNORECASE)
            value_text = number_match.group(0) if number_match else ""
            if not value_text:
                continue
            key = sentence_key(sentence)
            if key.strip():
                candidates.append(
                    {
                        "source": source,
                        "page": page,
                        "sentence": sentence.strip(),
                        "value": value_text,
                        "key": key,
                    }
                )

    grouped: Dict[str, List[Dict[str, Any]]] = {}
    for candidate in candidates:
        grouped.setdefault(candidate["key"], []).append(candidate)

    conflict_items: List[Dict[str, Any]] = []
    seen = set()
    for key, items in grouped.items():
        values = {item["value"].lower() for item in items}
        if len(values) > 1:
            for item in items:
                tag = (item["source"], item["page"], item["sentence"])
                if tag not in seen:
                    seen.add(tag)
                    conflict_items.append(item)
    return conflict_items


def ask_llm(question: str, context_docs: List[Document], api_key: str) -> str:
    """Generate an answer grounded in retrieved chunks only."""
    try:
        llm = ChatOpenAI(
            model="gpt-4o-mini",
            temperature=0,
            api_key=api_key,
        )

        context_parts = []
        for index, doc in enumerate(context_docs, start=1):
            source_name = get_safe_file_name(doc.metadata.get("source"))
            page_number = int(doc.metadata.get("page", 1) or 1)
            context_parts.append(f"[Document {index} | Source: {source_name} | Page {page_number}]\n{doc.page_content.strip()}")

        prompt = f"""
You are a careful answering assistant.

Rules:
1. Use only the source material provided below.
2. If the source content does not provide enough evidence, reply exactly:
   "I could not find sufficient information in the uploaded documents to answer this question reliably."
3. Never invent facts, page numbers, sources, dates, or policies.
4. If the sources disagree, clearly say that the documents contain conflicting information.
5. Do not claim one document is more authoritative unless the documents themselves provide a clear reason.

Question: {question}

Source passages:
{chr(10).join(context_parts)}
"""

        response = llm.invoke([
            SystemMessage(content="Answer using only the provided passages and give brief, source-grounded responses."),
            HumanMessage(content=prompt),
        ])
        return response.content.strip()
    except Exception as exc:
        raise RuntimeError("The LLM could not generate an answer. Please verify your OPENAI_API_KEY and try again.") from exc


def format_conflict_message(conflicts: List[Dict[str, Any]]) -> str:
    """Turn detected conflicts into a readable explanation for the user."""
    lines = ["The uploaded documents contain conflicting information relevant to your question."]
    for item in conflicts:
        lines.append(f"- {item['source']} — Page {item['page']}: {item['sentence']}")
    lines.append("Conclusion: The documents disagree. The app is surfacing the disagreement without claiming one source is automatically authoritative.")
    return "\n".join(lines)


def add_website_urls() -> None:
    """Queue URLs from the input and clear the widget through its callback."""
    raw_urls = [item.strip() for item in st.session_state.url_input.split(",") if item.strip()]
    if not raw_urls:
        st.session_state.url_input_feedback = ("warning", "Please enter at least one valid URL before adding it.")
        return

    normalized_urls = [
        url if re.match(r"^https?://", url, flags=re.IGNORECASE) else f"https://{url}"
        for url in raw_urls
    ]
    existing_urls = st.session_state.website_urls
    unique_urls = list(dict.fromkeys(existing_urls + normalized_urls))
    added_count = len(unique_urls) - len(existing_urls)
    st.session_state.website_urls = unique_urls
    st.session_state.url_input = ""
    st.session_state.url_input_feedback = ("success", f"Added {added_count} new URL(s) to the indexing queue.")


def app() -> None:
    st.set_page_config(page_title="Multi-Source RAG App", page_icon="📚")
    st.markdown(
        """
        <style>
            :root {
                --bg: #f3f7ff;
                --panel: rgba(255, 255, 255, 0.88);
                --panel-strong: #ffffff;
                --text: #1d2433;
                --muted: #5f6f8a;
                --primary: #3b82f6;
                --primary-soft: #eaf2ff;
                --primary-deep: #1d4ed8;
                --border: #dfeaf8;
                --shadow: 0 18px 45px rgba(59, 130, 246, 0.10);
            }

            .stApp {
                background: linear-gradient(180deg, #f8fbff 0%, #eef5ff 100%);
                color: var(--text);
            }

            .main .block-container {
                max-width: 1200px;
                padding-top: 2.25rem;
                padding-bottom: 2.5rem;
            }

            h1 {
                color: #13294b;
                letter-spacing: -0.03em;
                margin-bottom: 0.25rem;
            }

            .stButton > button {
                background: linear-gradient(135deg, var(--primary), var(--primary-deep));
                color: white;
                border: none;
                border-radius: 12px;
                padding: 0.68rem 1.15rem;
                font-weight: 700;
                box-shadow: 0 12px 26px rgba(59, 130, 246, 0.20);
            }

            .stButton > button:hover {
                box-shadow: 0 16px 28px rgba(59, 130, 246, 0.28);
                transform: translateY(-1px);
            }

            .stTextInput > div > div > input,
            .stTextArea > div > div > textarea {
                background: white;
                border: 1px solid var(--border);
                border-radius: 12px;
                color: var(--text);
                box-shadow: none;
            }

            [data-testid="stFileUploaderDropzone"] {
                border: 1.5px dashed #9dc2ff;
                border-radius: 18px;
                background: rgba(255, 255, 255, 0.76);
            }

            .stInfo, .stSuccess, .stWarning, .stError {
                border-radius: 12px;
                padding: 0.9rem 1rem;
                border: 1px solid #d8e2ef;
                background: #ffffff;
                color: #1f2937 !important;
            }

            [data-testid="stAlert"],
            [data-testid="stAlert"] [data-testid="stMarkdownContainer"],
            [data-testid="stAlert"] [data-testid="stMarkdownContainer"] p {
                color: #1f2937 !important;
            }

            .source-pill {
                display: inline-block;
                background: var(--primary-soft);
                color: var(--primary-deep);
                border: 1px solid #cfe0ff;
                border-radius: 999px;
                padding: 0.28rem 0.72rem;
                margin: 0.18rem 0.35rem 0.18rem 0;
                font-size: 0.82rem;
                font-weight: 600;
            }
        </style>
        """,
        unsafe_allow_html=True,
    )

    st.title("📚 Multi-Source RAG Assistant")
    st.caption("Upload PDFs and add website URLs to build a unified, searchable knowledge base.")

    api_key = get_openai_api_key()
    if not api_key:
        st.error("Missing OPENAI_API_KEY. Run: export OPENAI_API_KEY='your_key_here' before starting the app.")
        st.stop()

    if "vector_store" not in st.session_state:
        st.session_state.vector_store = None
    if "indexed_chunks" not in st.session_state:
        st.session_state.indexed_chunks = 0
    if "index_message" not in st.session_state:
        st.session_state.index_message = "No sources indexed yet."
    if "website_urls" not in st.session_state:
        st.session_state.website_urls = []
    if "last_upload_signature" not in st.session_state:
        st.session_state.last_upload_signature = None

    with st.sidebar:
        st.markdown("### Source Library")
        st.caption("Build your knowledge base from PDFs and web content.")

        uploaded_files = st.file_uploader(
            "Upload PDF files",
            type=["pdf"],
            accept_multiple_files=True,
            help="Choose multiple PDFs to add to the same vector database.",
        )

        if uploaded_files:
            st.write("Uploaded PDFs:")
            for file in uploaded_files:
                st.markdown(f'<span class="source-pill">{file.name}</span>', unsafe_allow_html=True)

        st.text_input(
            "Website URL (optional)",
            placeholder="https://example.com or example.com",
            help="Add a website URL to include in the vector DB alongside the PDFs.",
            key="url_input",
        )

        st.button("Add URL to sources", key="add_url_button", on_click=add_website_urls)
        url_input_feedback = st.session_state.pop("url_input_feedback", None)
        if url_input_feedback:
            feedback_type, feedback_message = url_input_feedback
            if feedback_type == "warning":
                st.warning(feedback_message)
            else:
                st.success(feedback_message)

        if st.session_state.website_urls:
            st.write("Website sources:")
            for idx, url in enumerate(st.session_state.website_urls):
                col1, col2 = st.columns([5, 2])
                with col1:
                    st.markdown(f'<span class="source-pill">{url}</span>', unsafe_allow_html=True)
                with col2:
                    if st.button("Remove", key=f"remove_url_{idx}"):
                        st.session_state.website_urls.pop(idx)
                        st.rerun()

            if st.button("Clear website sources", key="clear_web_sources"):
                st.session_state.website_urls = []
                st.success("Website sources cleared.")

        st.markdown("---")
        if st.button("Reset all sources", key="reset_all_sources"):
            st.session_state.website_urls = []
            st.session_state.vector_store = None
            st.session_state.indexed_chunks = 0
            st.session_state.index_message = "No sources indexed yet."
            st.success("All sources cleared.")

    st.markdown("### 1) Selected sources")
    if uploaded_files:
        st.write("PDFs in queue:")
        for file in uploaded_files:
            st.markdown(f'<span class="source-pill">{file.name}</span>', unsafe_allow_html=True)

    if st.session_state.website_urls:
        st.write("Website sources in queue:")
        for url in st.session_state.website_urls:
            st.markdown(f'<span class="source-pill">{url}</span>', unsafe_allow_html=True)
    else:
        st.write("No website URLs added yet.")

    st.markdown("---")

    if st.button("Index sources", key="index_sources_button"):
        if not uploaded_files and not st.session_state.website_urls:
            st.warning("Please upload at least one PDF or add at least one website URL before indexing.")
        else:
            with st.spinner("Loading and indexing all selected sources..."):
                all_documents: List[Document] = []
                valid_pdf_count = 0
                valid_url_count = 0

                for uploaded_file in uploaded_files or []:
                    if not uploaded_file.name.lower().endswith(".pdf"):
                        st.warning(f"{uploaded_file.name} is not a PDF and was skipped.")
                        continue

                    file_docs = read_pdf_documents(uploaded_file)
                    if file_docs:
                        valid_pdf_count += 1
                        all_documents.extend(file_docs)

                for url in st.session_state.website_urls:
                    url_docs = read_website_documents(url)
                    if url_docs:
                        valid_url_count += 1
                        all_documents.extend(url_docs)

                if not all_documents:
                    st.session_state.vector_store = None
                    st.session_state.indexed_chunks = 0
                    st.session_state.index_message = "No readable text was found in the uploaded PDFs or website URLs."
                    st.warning("No readable text was found in the uploaded PDFs or website URLs.")
                else:
                    try:
                        embeddings = OpenAIEmbeddings(model="text-embedding-3-small", api_key=api_key)
                        vector_store, chunk_count = build_vector_store(all_documents, embeddings)
                    except Exception:
                        st.error("The app could not index the sources because the OpenAI API key is invalid or the OpenAI service is unavailable. Please check your OPENAI_API_KEY and try again.")
                        st.session_state.vector_store = None
                        st.session_state.indexed_chunks = 0
                        st.session_state.index_message = "Indexing failed. Please verify the OPENAI_API_KEY."
                        st.stop()

                    st.session_state.vector_store = vector_store
                    st.session_state.indexed_chunks = chunk_count
                    st.session_state.index_message = (
                        f"Indexed {valid_pdf_count} PDF(s) and {valid_url_count} website(s) with {chunk_count} chunk(s)."
                    )
                    st.success(st.session_state.index_message)

    st.info(st.session_state.index_message)
    st.write(f"Indexed chunks: {st.session_state.indexed_chunks}")

    st.markdown("### 2) Ask a question")
    question = st.text_input(
        "Ask a question about the indexed sources",
        placeholder="Example: What is the probation period for new employees?",
        key="question_input",
    )

    if question and st.button("Generate answer"):
        if st.session_state.vector_store is None:
            st.warning("Please index at least one readable PDF or website before asking a question.")
        else:
            with st.spinner("Retrieving relevant passages and generating the answer..."):
                retrieved_docs = st.session_state.vector_store.similarity_search(question, k=3)

                if not retrieved_docs:
                    st.markdown("### Answer")
                    st.write("I could not find sufficient information in the uploaded documents to answer this question reliably.")
                    st.markdown("### Sources")
                    st.write("No relevant supporting source was found.")
                    return

                conflict_items = detect_conflicts(question, retrieved_docs)
                try:
                    answer_text = ask_llm(question, retrieved_docs, api_key)
                except Exception:
                    st.error("The app could not generate an answer because the OpenAI API key is invalid or the service is unavailable. Please check your OPENAI_API_KEY and try again.")
                    st.stop()

                if conflict_items:
                    answer_text = format_conflict_message(conflict_items)

                citations = unique_citations(retrieved_docs)

                st.markdown("### Answer")
                st.write(answer_text)

                st.markdown("### Sources")
                if citations:
                    for citation in citations:
                        st.write(f"- {citation['filename']} — Page {citation['page']}")
                else:
                    st.write("No relevant supporting source was found.")

                if conflict_items:
                    st.markdown("### Conflicting statements")
                    for item in conflict_items:
                        st.write(f"- {item['source']} — Page {item['page']}: {item['sentence']}")


if __name__ == "__main__":
    app()
