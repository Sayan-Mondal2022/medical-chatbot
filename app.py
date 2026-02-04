import os
import streamlit as st
from dotenv import load_dotenv

from src.helpers import (
    load_pdf_file,
    filter_to_minimal_docs,
    text_split,
    download_embeddings_model
)
from src.prompt import system_prompt

from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_pinecone import PineconeVectorStore


# To load the API keys from .env file
load_dotenv()
PINECONE_API_KEY = os.getenv("PINECONE_API")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

os.environ["PINECONE_API_KEY"] = PINECONE_API_KEY
os.environ["GROQ_API_KEY"] = GROQ_API_KEY

INDEX_NAME = "medical-chatbot"

# ---------------------------------
# STREAMLIT UI
st.set_page_config(page_title="🩺 Medical Chatbot", layout="centered")
st.title("🩺 Medical RAG Chatbot")

# ---------------------------------
# PROMPT
prompt = ChatPromptTemplate.from_messages([
    ("system", system_prompt),
    ("human", "{input}")
])

# ---------------------------------
# LOAD EMBEDDINGS (cached)
@st.cache_resource
def load_embeddings():
    return download_embeddings_model()

embeddings = load_embeddings()

# ---------------------------------
# LOAD EXISTING VECTOR STORE (NO INDEXING)
@st.cache_resource
def load_vectorstore():
    return PineconeVectorStore.from_existing_index(
        index_name=INDEX_NAME,
        embedding=embeddings
    )

docsearch = load_vectorstore()

retriever = docsearch.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 5}
)

# ---------------------------------
# LLM
chatModel = ChatGroq(
    model="llama-3.3-70b-versatile",
    temperature=0.2
)

# ---------------------------------
# RAG CHAIN
rag_chain = (
    {
        "context": retriever | (lambda docs: "\n\n".join(d.page_content for d in docs)),
        "input": RunnablePassthrough()
    }
    | prompt
    | chatModel
    | StrOutputParser()
)

# ---------------------------------
# PDF UPLOAD (INDEX ONLY HERE)
st.sidebar.header("📄 Upload PDF")

uploaded_file = st.sidebar.file_uploader(
    "Upload a medical PDF",
    type=["pdf"]
)

if uploaded_file and st.sidebar.button("Add to Knowledge Base"):
    with st.spinner("Processing and indexing PDF..."):
        os.makedirs("data", exist_ok=True)

        file_path = os.path.join("data", uploaded_file.name)
        with open(file_path, "wb") as f:
            f.write(uploaded_file.read())

        extracted_pdf = load_pdf_file("data")
        minimal_docs = filter_to_minimal_docs(extracted_pdf)
        text_chunks = text_split(minimal_docs)

        docsearch.add_documents(text_chunks)

    st.sidebar.success("✅ PDF indexed successfully")

# ---------------------------------
# CHAT HISTORY
if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# ---------------------------------
# USER INPUT
query = st.chat_input("Ask a medical question...")

if query:
    st.session_state.messages.append(
        {"role": "user", "content": query}
    )
    with st.chat_message("user"):
        st.markdown(query)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            response = rag_chain.invoke(query)
            st.markdown(response)

    st.session_state.messages.append(
        {"role": "assistant", "content": response}
    )
