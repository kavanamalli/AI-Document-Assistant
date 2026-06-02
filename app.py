from dotenv import load_dotenv
import os

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_google_genai import ChatGoogleGenerativeAI


# ==========================
# LOAD ENVIRONMENT VARIABLES
# ==========================

load_dotenv()

api_key = os.getenv("GOOGLE_API_KEY")

if not api_key:
    raise ValueError(
        "GOOGLE_API_KEY not found in .env file"
    )


# ==========================
# LOAD PDF
# ==========================

loader = PyPDFLoader(
    r"D:\AI DOCUMENT\pdf\Kavana_internship_report.pdf"
)

documents = loader.load()
for i, doc in enumerate(documents):
    doc.metadata["page"] = i


print(f"\nPages Loaded: {len(documents)}")


# ==========================
# CHUNKING
# ==========================

splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200
)

chunks = splitter.split_documents(documents)

print(f"Number of Chunks: {len(chunks)}")


# ==========================
# EMBEDDINGS
# ==========================

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-mpnet-base-v2"
)

print("Embedding Model Loaded")


# ==========================
# VECTOR DATABASE
# ==========================

vectorstore = Chroma.from_documents(
    documents=chunks,
    embedding=embeddings,
    persist_directory="./chroma_db"
)

print("Vector Database Created")


# ==========================
# GEMINI MODEL
# ==========================

llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    google_api_key=api_key
)

print("Gemini Model Loaded")
chat_history = []

# ==========================
# QUESTION ANSWERING LOOP
# ==========================

while True:

    question = input("\nAsk a question (or type 'exit'): ")

    if question.lower() == "exit":
        break

    retriever = vectorstore.as_retriever(
        search_type="mmr",
        search_kwargs={
            "k": 8,
            "fetch_k": 20
        }
    )

    docs = retriever.invoke(question)

    first_page = documents[0].page_content

    context = first_page + "\n\n"

    for doc in docs:
        context += f"""
Page {doc.metadata.get('page')}:

{doc.page_content}

"""

    history = ""

    for chat in chat_history[-3:]:
        history += f"""
Question: {chat['question']}
Answer: {chat['answer']}
"""

    prompt = f"""
You are an intelligent document assistant.

Previous Conversation:
{history}

Context:
{context}

Question:
{question}

Answer:
"""

    response = llm.invoke(prompt)

    chat_history.append({
        "question": question,
        "answer": response.content
    })
    chat_history = chat_history[-5:]

    print("\n" + "=" * 60)
    print("ANSWER")
    print("=" * 60)

    print(response.content)

    print("\n" + "=" * 60)