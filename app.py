import os
import tempfile

import streamlit as st
from dotenv import load_dotenv

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma

from langchain_google_genai import (
    GoogleGenerativeAIEmbeddings,
    ChatGoogleGenerativeAI,
)

from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate


# =========================================================
# Load environment variables
# =========================================================

load_dotenv()


# =========================================================
# Streamlit configuration
# =========================================================

st.set_page_config(
    page_title="DocuLens | Grounded PDF Assistant",
    page_icon="📑",
    layout="wide",
)


# =========================================================
# Title
# =========================================================

st.title("📑 DocuLens: Grounded PDF Assistant")

st.markdown(
    "Ask questions about complex documents and receive answers "
    "with **exact page citations**."
)


# =========================================================
# Sidebar
# =========================================================

with st.sidebar:

    st.header("⚙️ Configuration")

    user_api_key = st.text_input(
        "Gemini API Key",
        type="password",
        value=os.getenv("GEMINI_API_KEY", ""),
        help="Enter your Google Gemini API key.",
    )

    st.divider()

    uploaded_file = st.file_uploader(
        "Upload a PDF document",
        type=["pdf"],
    )

    st.divider()

    st.markdown(
        """
        **Tech Stack**

        - Streamlit
        - LangChain
        - Google Gemini
        - Gemini Embeddings
        - ChromaDB
        - PyPDF
        """
    )


# =========================================================
# API key check
# =========================================================

if not user_api_key:

    st.warning(
        "⚠️ Please provide a Gemini API key in the sidebar "
        "to run the app."
    )

    st.stop()


# Make API key available to LangChain
os.environ["GOOGLE_API_KEY"] = user_api_key


# =========================================================
# Process PDF
# =========================================================

@st.cache_resource(show_spinner=False)
def process_pdf(file_bytes):

    progress = st.empty()

    progress.info("📄 Loading PDF...")

    # -----------------------------------------------------
    # Create temporary PDF
    # -----------------------------------------------------

    with tempfile.NamedTemporaryFile(
        delete=False,
        suffix=".pdf",
    ) as tmp:

        tmp.write(file_bytes)
        tmp_path = tmp.name

    try:

        # -------------------------------------------------
        # Load PDF
        # -------------------------------------------------

        loader = PyPDFLoader(tmp_path)

        docs = loader.load()

        if not docs:
            raise ValueError(
                "The PDF does not contain readable text."
            )

        progress.info(
            f"📄 Loaded {len(docs)} page(s). Splitting document..."
        )

        # -------------------------------------------------
        # Split document
        # -------------------------------------------------

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=150,
            separators=[
                "\n\n",
                "\n",
                " ",
                "",
            ],
        )

        chunks = splitter.split_documents(docs)

        if not chunks:
            raise ValueError(
                "No readable text chunks were created from the PDF."
            )

        # -------------------------------------------------
        # Limit extremely large documents
        # -------------------------------------------------

        MAX_CHUNKS = 500

        if len(chunks) > MAX_CHUNKS:

            st.warning(
                f"⚠️ This PDF produced {len(chunks)} chunks. "
                f"Only the first {MAX_CHUNKS} chunks will be indexed "
                "to keep API usage reasonable."
            )

            chunks = chunks[:MAX_CHUNKS]

        progress.info(
            f"🧠 Creating embeddings for {len(chunks)} chunks..."
        )

        # -------------------------------------------------
        # Gemini Embeddings
        # -------------------------------------------------

        embeddings = GoogleGenerativeAIEmbeddings(
            model="models/gemini-embedding-001",
            task_type="RETRIEVAL_DOCUMENT",
        )

        # -------------------------------------------------
        # Create Chroma vector database
        # -------------------------------------------------

        vector_store = Chroma.from_documents(
            documents=chunks,
            embedding=embeddings,
        )

        progress.success(
            "✅ PDF indexed successfully."
        )

        return vector_store

    finally:

        # -------------------------------------------------
        # Delete temporary PDF
        # -------------------------------------------------

        if os.path.exists(tmp_path):

            os.remove(tmp_path)


# =========================================================
# Main application
# =========================================================

if uploaded_file:

    try:

        # -------------------------------------------------
        # Process PDF
        # -------------------------------------------------

        vector_store = process_pdf(
            uploaded_file.getvalue()
        )

        # -------------------------------------------------
        # Retriever
        # -------------------------------------------------

        retriever = vector_store.as_retriever(
            search_kwargs={
                "k": 4
            }
        )

        # -------------------------------------------------
        # System prompt
        # -------------------------------------------------

        system_prompt = """
You are an expert document analysis assistant.

Answer the user's question strictly using ONLY the
provided document context.

Rules:

1. Do not invent information.
2. Do not use outside knowledge.
3. If the answer is not present in the document,
   clearly say that the information is not available
   in the provided document.
4. Give concise and accurate answers.
5. Use the retrieved context carefully.

Context:

{context}
"""

        prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    system_prompt,
                ),
                (
                    "human",
                    "{input}",
                ),
            ]
        )

        # -------------------------------------------------
        # Gemini Chat Model
        # -------------------------------------------------

        llm = ChatGoogleGenerativeAI(
            model="gemini-2.5-flash",
            temperature=0,
        )

        # -------------------------------------------------
        # Document combination chain
        # -------------------------------------------------

        combine_chain = create_stuff_documents_chain(
            llm,
            prompt,
        )

        # -------------------------------------------------
        # Retrieval chain
        # -------------------------------------------------

        rag_chain = create_retrieval_chain(
            retriever,
            combine_chain,
        )

        # -------------------------------------------------
        # Chat history
        # -------------------------------------------------

        if "messages" not in st.session_state:

            st.session_state.messages = []

        # Display previous messages

        for message in st.session_state.messages:

            with st.chat_message(
                message["role"]
            ):

                st.markdown(
                    message["content"]
                )

        # -------------------------------------------------
        # User question
        # -------------------------------------------------

        query = st.chat_input(
            "Ask a question about your document..."
        )

        if query:

            # -------------------------------------------------
            # Display user question
            # -------------------------------------------------

            with st.chat_message("user"):

                st.markdown(query)

            st.session_state.messages.append(
                {
                    "role": "user",
                    "content": query,
                }
            )

            # -------------------------------------------------
            # Generate answer
            # -------------------------------------------------

            with st.chat_message("assistant"):

                with st.spinner(
                    "🔎 Searching document..."
                ):

                    try:

                        response = rag_chain.invoke(
                            {
                                "input": query
                            }
                        )

                        answer = response.get(
                            "answer",
                            "I could not find an answer in the document.",
                        )

                        # -------------------------------------------------
                        # Extract page numbers
                        # -------------------------------------------------

                        pages = sorted(
                            {
                                doc.metadata.get(
                                    "page",
                                    0
                                ) + 1
                                for doc in response.get(
                                    "context",
                                    []
                                )
                            }
                        )

                        if pages:

                            citation_text = (
                                "\n\n"
                                "**📍 Source Citations:** "
                                f"Page(s) "
                                f"{', '.join(map(str, pages))}"
                            )

                        else:

                            citation_text = (
                                "\n\n"
                                "**📍 Source Citations:** "
                                "No page information available."
                            )

                        full_response = (
                            answer
                            + citation_text
                        )

                        st.markdown(
                            full_response
                        )

                        # Store assistant response

                        st.session_state.messages.append(
                            {
                                "role": "assistant",
                                "content": full_response,
                            }
                        )

                    except Exception as e:

                        error_text = str(e)

                        if (
                            "429" in error_text
                            or "rate" in error_text.lower()
                            or "quota" in error_text.lower()
                        ):

                            st.error(
                                "⚠️ Gemini API rate limit or quota "
                                "was reached. Please wait a moment "
                                "and try again."
                            )

                        else:

                            st.error(
                                "❌ An error occurred while "
                                "generating the answer."
                            )

                        st.code(
                            error_text,
                            language="text",
                        )

    except Exception as e:

        error_text = str(e)

        if (
            "429" in error_text
            or "rate" in error_text.lower()
            or "quota" in error_text.lower()
        ):

            st.error(
                "⚠️ Gemini API rate limit or quota reached "
                "while creating the document embeddings."
            )

            st.info(
                "Try a smaller PDF or wait before uploading "
                "the document again."
            )

        else:

            st.error(
                "❌ An error occurred while processing the PDF."
            )

        st.code(
            error_text,
            language="text",
        )

else:

    st.info(
        "👈 Upload a PDF in the sidebar to start querying."
    )