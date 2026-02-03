import os
from dotenv import load_dotenv
from src.helpers import load_pdf_file, filter_to_minimal_docs, text_split, download_embeddings_model
from pinecone import Pinecone
from pinecone import ServerlessSpec
from langchain_pinecone import PineconeVectorStore


# To load the API keys from .env file
load_dotenv()
PINECONE_API_KEY = os.getenv("PINECONE_API")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# Setting up environment variables
os.environ["PINECONE_API_KEY"] = PINECONE_API_KEY
os.environ["GROQ_API_KEY"] = GROQ_API_KEY

# Loading and processing the PDF files
print("Extracting the Data from the PDF...")
extracted_pdf = load_pdf_file("data")
print("Extraction is Completed\n")

# Filtering to minimal documents
print("\nFiltering the Extracted DOCs...")
minimal_docs = filter_to_minimal_docs(extracted_pdf)
print("Filtering has been completed\n")

# Chunking the documents
print("\nPerforming chunking on Filtered Docs")
text_chunks = text_split(minimal_docs)
print("Chunking is completed\n")


# Loading the Embeddings model
print("\nLoading the Embedding model...")
embeddings = download_embeddings_model()
print("Embedding model has been loaded\n")


# Authenticate Pinecone Client
print("\nAuthenticating with the PineCone Vector DB")
pc = Pinecone(api_key=PINECONE_API_KEY)

# Setting up Pinecone Index
print("\nCreating the Pinecone Index...")
index_name = "medical-chatbot"

if not pc.has_index(index_name):
    pc.create_index(
        name=index_name,
        dimension=384,  # Dimension for all-MiniLM-L6-v2 embeddings
        metric="cosine",
        spec=ServerlessSpec(
            cloud="aws",
            region="us-east-1",
            )
    )
index = pc.Index(index_name)
print("PineCone Index has been created...")

# 1. This function will take the text chunks 
# 2. Generate embeddings using the HuggingFaceEmbeddings model
# 3. Store the embeddings in the Pinecone index
print("Storing the Vector Embeddings into my Database")
docsearch = PineconeVectorStore.from_documents(
    documents=text_chunks,
    embedding=embeddings,
    index_name=index_name,
)
print("Storing has been completed")