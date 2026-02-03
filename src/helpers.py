from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from typing import List
from langchain_core.documents import Document
from langchain_community.embeddings import HuggingFaceEmbeddings

# This function loads PDF files from a specified directory
def load_pdf_file(file_path):
    loader = DirectoryLoader(
        file_path,
        glob="*.pdf",
        loader_cls=PyPDFLoader,
    )
    documents = loader.load()
    return documents


def filter_to_minimal_docs(docs: List[Document]) -> List[Document]:
    """
    Filters documents to only include page_content and source metadata.
    """
    
    minimal_docs = []
    for doc in docs:
        minimal_doc = Document(
            page_content=doc.page_content,
            metadata={"source": doc.metadata.get("source", "")}
        )
        minimal_docs.append(minimal_doc)

    return minimal_docs


# Split the documents into smaller chunks
def text_split(minimal_docs: List[Document]) -> List[Document]:
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=20
    )
    split_docs = text_splitter.split_documents(minimal_docs)
    return split_docs


# Using sentence Transformers model for generating embeddings
def download_embeddings_model(
        model_name = "sentence-transformers/all-MiniLM-L6-v2"
    ):
    return HuggingFaceEmbeddings(model_name=model_name)