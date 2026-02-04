# 🩺 Medical Chatbot (FAQ-based Healthcare Assistant)

The **Medical Chatbot** is an AI-powered healthcare assistant designed to answer frequently asked medical questions related to common health conditions and diseases.
It leverages **Retrieval-Augmented Generation (RAG)** to provide **context-aware, document-grounded responses**, helping users gain **basic guidance and temporary relief suggestions**.

> **⚠️ Disclaimer**:\
> This chatbot is **not a replacement for professional medical advice**. It is intended for **educational and informational purposes only**. Users are strongly advised to consult certified medical professionals for diagnosis and treatment.

## 🎯 Key Objectives:

- Provide quick and reliable answers to common medical FAQs
- Reduce misinformation by grounding responses in trusted documents
- Demonstrate real-world use of **RAG pipelines in healthcare AI**
- Build a scalable and modular chatbot architecture


## ✨ Features

- 📄 Document-based Question Answering
- 🧠 Context-aware responses using RAG
- 🔍 Semantic search with vector embeddings
- ⚡ Fast and interactive UI using Streamlit
- 🔐 Secure API key handling using environment variables
- 🧩 Modular and extensible architecture

## 🧰 Tech Stack

- **Python** – Core programming language
- **Streamlit** – Interactive web interface
- **LangChain** – LLM orchestration and RAG pipeline
- **Pinecone** – Vector database for semantic search
- **RAG (Retrieval-Augmented Generation)** – Accurate and grounded responses

## 🧠 How It Works (Architecture Overview)

1. Medical documents are converted into vector embeddings
2. Embeddings are stored in **Pinecone Vector Database**
3. User queries are embedded and matched semantically
4. Relevant document chunks are retrieved
5. A Large language model (LLM) generates responses using retrieved context

This ensures **less hallucination and more reliable** answers compared to standard LLM chatbots.

## ⚙️ Setup Instructions

### 1. Creating the Environment with specified python version
```bash
# Using Python version 3.11.x
py -3.11 -m venv .venv

# Then activate the Python Virtual env
.venv\Scripts\activate
```

### 2. Cloning the GitHub repo
```bash
git clone https://github.com/Sayan-Mondal2022/medical-chatbot.git

# In the Conda terminal navigate to your project folder
cd medical-chatbot
```

### 3. Install all the dependencies
```bash
pip install -r requirements.txt
```

### 🔑 Environment Variables
Create a `.env` file in the root directory and add:
```bash
PINECONE_API_KEY=your_pinecone_api_key
OPENAI_API_KEY=your_openai_api_key
```

### 4. Running the Application
```bash
streamlit run app.py
```
The app will launch locally in your browser.

## 🙏 Acknowledgement

I sincerely thank the open-source community and the developers of Streamlit, LangChain, and Pinecone for providing the tools and documentation that made this project possible.

I also acknowledge the book used as a knowledge source for generating vector embeddings, which played a key role in building the chatbot’s retrieval system.

**Book Reference:**

*The Gale Encyclopedia of Medicine* (2nd ed.). Gale Group.
Available at: https://www.academia.edu/32752835/The_GALE_ENCYCLOPEDIA_of_MEDICINE_SECOND_EDITION

## 💙 Thank You

Thank you for taking the time to explore this project.
Your feedback and suggestions are always welcome, and your support is truly appreciated.
