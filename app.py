import os
import tempfile
import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv

load_dotenv()

st.set_page_config(
    page_title="DocuLens | Grounded PDF Assistant",
    page_icon="📑",
    layout="wide"
)

st.title("📑 DocuLens: Grounded PDF Assistant")
st.markdown("Ask questions about complex documents and receive answers with **exact page citations**.")

# Sidebar for configuration
with st.sidebar:
    st.header("⚙️ Configuration")
    user_api_key = st.text_input(
        "OpenAI API Key",
        type="password",
        value=os.getenv("OPENAI_API_KEY", ""),
        help="Enter your OpenAI API key or set it in your .env / Streamlit secrets"
    )
    st.divider()
    uploaded_file = st.file_uploader("Upload a PDF document", type=["pdf"])
    st.markdown("---")
    st.markdown(
        "**Tech Stack:**\n"
        "- Streamlit\n"
        "- LangChain\n"
        "- OpenAI (GPT-4o-mini & Text-Embedding-3)\n"
        "- ChromaDB\n"
        "- PyPDF"
    )

if not user_api_key:
    st.warning("⚠️ Please provide an OpenAI API key in the sidebar to run the app.")
    st.stop()

os.environ["OPENAI_API_KEY"] = user_api_key

@st.cache_resource(show_spinner="📄 Chunking, embedding, and indexing PDF...")
def process_pdf(file_bytes):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(file_bytes)
        tmp_path = tmp.name

    loader = PyPDFLoader(tmp_path)
    docs = loader.load()

    # Smart recursive splitting
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=120,
        separators=["\n\n", "\n", " ", ""]
    )
    chunks = splitter.split_documents(docs)

    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    vector_store = Chroma.from_documents(chunks, embeddings)

    os.remove(tmp_path)
    return vector_store

if uploaded_file:
    vector_store = process_pdf(uploaded_file.getvalue())
    retriever = vector_store.as_retriever(search_kwargs={"k": 4})

    system_prompt = (
        "You are an expert document analysis assistant. Answer the user question strictly using "
        "only the provided context snippets. If the answer cannot be deduced from the context, "
        "explicitly state that the information is missing from the document.\n\n"
        "Context:\n{context}"
    )

    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "{input}"),
    ])

    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0, streaming=True)
    combine_chain = create_stuff_documents_chain(llm, prompt)
    rag_chain = create_retrieval_chain(retriever, combine_chain)

    # Chat session state
    if "messages" not in st.session_state:
        st.session_state.messages = []

    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    if query := st.chat_input("Ask a question about your document..."):
        st.session_state.messages.append({"role": "user", "content": query})
        with st.chat_message("user"):
            st.markdown(query)

        with st.chat_message("assistant"):
            with st.spinner("Searching document & synthesizing answer..."):
                response = rag_chain.invoke({"input": query})
                answer = response["answer"]

                # Extract and deduplicate source page numbers
                pages = sorted(list({
                    doc.metadata.get("page", 0) + 1 for doc in response["context"]
                }))
                citation_text = f"\n\n**📍 Source Citations:** Page(s) {', '.join(map(str, pages))}"
                full_response = answer + citation_text
                st.markdown(full_response)

        st.session_state.messages.append({"role": "assistant", "content": full_response})
else:
    st.info("👈 Upload a PDF in the sidebar to start querying.")
