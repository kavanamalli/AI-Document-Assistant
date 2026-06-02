from dotenv import load_dotenv
import os
import tempfile

import streamlit as st

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_google_genai import ChatGoogleGenerativeAI


# ==========================
# PAGE CONFIG
# ==========================

st.set_page_config(
    page_title="AI Document Assistant",
    page_icon="📄"
)

st.title("📄 AI Document Assistant")
st.write("Upload any PDF and ask questions about it.")


# ==========================
# ENV VARIABLES
# ==========================

load_dotenv()

api_key = os.getenv("GOOGLE_API_KEY")

if not api_key:
    st.error("GOOGLE_API_KEY not found")
    st.stop()


# ==========================
# GEMINI MODEL
# ==========================

llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    google_api_key=api_key
)


# ==========================
# SESSION STATE
# ==========================

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None

if "documents" not in st.session_state:
    st.session_state.documents = None

if "current_file" not in st.session_state:
    st.session_state.current_file = None


# ==========================
# FILE UPLOAD
# ==========================

uploaded_file = st.file_uploader(
    "Upload PDF",
    type=["pdf"]
)


# ==========================
# PROCESS PDF ONLY ONCE
# ==========================

if uploaded_file:

    if st.session_state.current_file != uploaded_file.name:

        with st.spinner("Processing PDF..."):

            with tempfile.NamedTemporaryFile(
                delete=False,
                suffix=".pdf"
            ) as tmp_file:

                tmp_file.write(uploaded_file.read())
                pdf_path = tmp_file.name

            # LOAD PDF

            loader = PyPDFLoader(pdf_path)
            documents = loader.load()

            for i, doc in enumerate(documents):
                doc.metadata["page"] = i + 1

            # CHUNKING

            splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000,
                chunk_overlap=200
            )

            chunks = splitter.split_documents(documents)

            # EMBEDDINGS

            embeddings = HuggingFaceEmbeddings(
                model_name="sentence-transformers/all-mpnet-base-v2"
            )

            # VECTOR DATABASE

            vectorstore = Chroma.from_documents(
                documents=chunks,
                embedding=embeddings
            )

            # SAVE IN SESSION

            st.session_state.vectorstore = vectorstore
            st.session_state.documents = documents
            st.session_state.current_file = uploaded_file.name
            st.session_state.chat_history = []

        st.success(
            f"PDF Loaded Successfully ({len(documents)} pages)"
        )


# ==========================
# CHAT INTERFACE
# ==========================

if st.session_state.vectorstore is not None:

    st.success(
        f"PDF Loaded Successfully ({len(st.session_state.documents)} pages)"
    )

    # DISPLAY CHAT HISTORY

    for chat in st.session_state.chat_history:

        with st.chat_message("user"):
            st.write(chat["question"])

        with st.chat_message("assistant"):
            st.write(chat["answer"])

            if chat.get("pages"):
                st.caption(
                    "📄 Source Pages: "
                    + ", ".join(map(str, chat["pages"]))
                )

    # QUESTION INPUT

    question = st.chat_input(
        "Ask a question about the document..."
    )

    if question:

        with st.chat_message("user"):
            st.write(question)

        retriever = st.session_state.vectorstore.as_retriever(
            search_type="mmr",
            search_kwargs={
                "k": 8,
                "fetch_k": 20
            }
        )

        docs = retriever.invoke(question)

        first_page = st.session_state.documents[0].page_content

        context = first_page + "\n\n"

        for doc in docs:

            context += f"""
Page {doc.metadata.get('page')}:

{doc.page_content}

"""

        # CHAT HISTORY FOR CONTEXT

        history = ""

        for chat in st.session_state.chat_history[-5:]:

            history += f"""
Question: {chat['question']}
Answer: {chat['answer']}
"""

        prompt = f"""
You are an intelligent document assistant.

Use the document context and conversation history
to answer accurately.

If the answer is not available in the document,
say:

"I could not find that information in the document."

Previous Conversation:
{history}

Context:
{context}

Question:
{question}

Answer:
"""

        with st.spinner("Generating answer..."):

            response = llm.invoke(prompt)

            answer = response.content

        pages = sorted(
            list(
                set(
                    doc.metadata.get("page", 0)
                    for doc in docs
                )
            )
        )

        with st.chat_message("assistant"):

            st.write(answer)

            st.caption(
                "📄 Source Pages: "
                + ", ".join(map(str, pages))
            )

        st.session_state.chat_history.append(
            {
                "question": question,
                "answer": answer,
                "pages": pages
            }
        )