# 📄 AI Document Assistant

An AI-powered document question-answering application built using Streamlit, LangChain, ChromaDB, Hugging Face Embeddings, and Google's Gemini LLM.

Users can upload PDF documents and interact with them using natural language questions. The system retrieves the most relevant content from the document and generates accurate answers using Retrieval-Augmented Generation (RAG).

## 🚀 Features

* Upload PDF documents
* Ask questions in natural language
* Retrieval-Augmented Generation (RAG)
* Semantic search using vector embeddings
* Conversational chat history
* Source page tracking
* Gemini-powered answer generation
* Streamlit web interface

## 🛠️ Tech Stack

* Python
* Streamlit
* LangChain
* ChromaDB
* Hugging Face Embeddings
* Google Gemini API
* PyPDF

## 📂 Project Workflow

1. User uploads a PDF.
2. PDF is loaded and split into chunks.
3. Text chunks are converted into embeddings.
4. Embeddings are stored in ChromaDB.
5. User asks a question.
6. Relevant chunks are retrieved using semantic search.
7. Gemini generates an answer based on the retrieved context.
8. Source page numbers are displayed.

## 📸 Application Screenshot

Add your screenshot here:

![Application Screenshot](images/app_screenshot.png)

## ⚙️ Installation

Clone the repository:

```bash
git clone https://github.com/your-username/AI-Document-Assistant.git
cd AI-Document-Assistant
```

Create a virtual environment:

```bash
python -m venv venv
```

Activate virtual environment:

```bash
venv\Scripts\activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Create a `.env` file:

```env
GOOGLE_API_KEY=YOUR_GEMINI_API_KEY
```

Run the application:

```bash
streamlit run streamlit.py
```

## 📈 Future Improvements

* Multi-PDF support
* Chat with multiple documents
* OCR for scanned PDFs
* Hybrid search (BM25 + Semantic Search)
* Citation highlighting
* PDF summarization
* Document comparison

