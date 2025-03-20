import os
import time
import streamlit as st
from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import FAISS
from groq import Groq

# Initialize Groq Client
GROQ_API_KEY = "your_groq_api_key"  # Replace with your actual API key
client = Groq(api_key=GROQ_API_KEY)

# Set up Streamlit page
st.set_page_config(page_title="HEC Pakistan Assistant", layout="wide")
st.title("📘 HEC Pakistan Assistant")

# Upload Folder
UPLOAD_FOLDER = "./data"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Function to load documents
def load_documents(uploaded_files):
    documents = []
    for file in uploaded_files:
        file_path = os.path.join(UPLOAD_FOLDER, file.name)
        
        with open(file_path, "wb") as f:
            f.write(file.read())

        if file.name.endswith(".pdf"):
            loader = PyPDFLoader(file_path)
        elif file.name.endswith(".docx"):
            loader = Docx2txtLoader(file_path)
        else:
            continue  

        documents.extend(loader.load())

    return documents

# Split documents into chunks
def split_documents(documents):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1500,
        chunk_overlap=300
    )
    return text_splitter.split_documents(documents)

# Initialize FAISS Vector Store
def create_faiss_db(documents):
    embeddings = OllamaEmbeddings(model="nomic-embed-text")
    db = FAISS.from_documents(documents, embedding=embeddings)
    return db

# Groq API Response
def get_groq_response(query, context):
    prompt = f"""
You are an AI assistant for HEC Pakistan. Answer queries based on the context.

**Context:** {context}

**User Query:** {query}

**HEC Assistant Response:**
"""
    
    response = client.chat.completions.create(
        messages=[{"role": "user", "content": prompt}],
        model="llama3-70b-8192",
        temperature=0.3,
        max_tokens=1024,
        stream=False,
    )
    
    return response.choices[0].message.content.strip()

# Streamlit UI
st.write("Upload documents related to HEC Pakistan and ask queries about policies, programs, and services.")

# File uploader
uploaded_files = st.file_uploader("Upload PDF/DOCX files", type=["pdf", "docx"], accept_multiple_files=True)

# User input
user_query = st.text_input("Ask HEC Assistant", placeholder="Type your query here...")

if st.button("Get Response"):
    if not uploaded_files:
        st.warning("⚠️ Please upload relevant documents first!")
    else:
        # Process files
        documents = load_documents(uploaded_files)
        chunks = split_documents(documents)
        db = create_faiss_db(chunks)

        # Get relevant context
        results = db.similarity_search(user_query, k=5)
        context = "\n".join([doc.page_content for doc in results])

        # Get AI response
        response = get_groq_response(user_query, context)

        # Show Response
        st.subheader("HEC Assistant Response:")
        st.write(response)
