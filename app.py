import os
from dotenv import load_dotenv
from src.helpers import load_pdf_file, filter_to_minimal_docs, text_split, download_embeddings_model
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_pinecone import PineconeVectorStore


# To load the API keys from .env file
load_dotenv()
PINECONE_API_KEY = os.getenv("PINECONE_API")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# Setting up environment variables
os.environ["PINECONE_API_KEY"] = PINECONE_API_KEY
os.environ["GROQ_API_KEY"] = GROQ_API_KEY

