import os
import time
import logging
import streamlit as st
from dotenv import load_dotenv
from langchain.text_splitter import RecursiveCharacterTextSplitter
from PyPDF2 import PdfReader
from docx import Document
from pinecone import Pinecone, ServerlessSpec
from uuid import uuid4
from io import BytesIO
from openai import OpenAI

# Load environment variables
load_dotenv()

# Pinecone setup
PINECONE_API_KEY = st.secrets["PINECONE_API_KEY"]
pc = Pinecone(api_key=PINECONE_API_KEY)

# OpenAI setup
OPENAI_API_KEY = st.secrets["CLIENT_KEY"]
openai_client = OpenAI(api_key=OPENAI_API_KEY)

INDEX_NAME = "bamb"
DIMENSION = 1536  # OpenAI 'text-embedding-3-small' returns 1536-dim vectors
REGION = "us-east-1"

# Create index if not already created
if INDEX_NAME not in pc.list_indexes().names():
    pc.create_index(
        name=INDEX_NAME,
        dimension=DIMENSION,
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", region=REGION)
    )

index = pc.Index(INDEX_NAME)

CHUNK_SIZE = 2000
CHUNK_OVERLAP = 500
logging.basicConfig(level=logging.INFO)

# Readers
def read_pdf(file):
    file.seek(0)
    pdf_reader = PdfReader(BytesIO(file.read()))
    return "\n".join([p.extract_text() for p in pdf_reader.pages if p.extract_text()])

def read_docx(file):
    file.seek(0)
    doc = Document(BytesIO(file.read()))
    return "\n".join([p.text for p in doc.paragraphs if p.text.strip()])

# Chunking
def chunk_document(text):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        length_function=len,
        separators=["\n\n", "\n", " ", ""]
    )
    return splitter.split_text(text)

# Generate OpenAI embeddings
def generate_embeddings(text_chunks):
    embeddings = []
    for i, chunk in enumerate(text_chunks):
        try:
            response = openai_client.embeddings.create(
                model="text-embedding-3-small", input=chunk
            )
            embedding = response.data[0].embedding
            embeddings.append(embedding)
        except Exception as e:
            logging.error(f"[ERROR] Embedding failed on chunk {i}: {e}")
            st.error(f"Embedding failed for chunk {i}.")
    return embeddings

# Store to Pinecone
def store_in_pinecone(chunks, embeddings):
    print(f"Chunks count: {len(chunks)}, Embeddings count: {len(embeddings)}")
    vectors = []
    for chunk, emb in zip(chunks, embeddings):
        if not isinstance(emb, list):
            emb = emb.tolist()  # convert if numpy array
        if len(emb) != DIMENSION:
            print(f"Skipping embedding with wrong dimension: {len(emb)}")
            continue
        vectors.append({"id": str(uuid4()), "values": emb, "metadata": {"text": chunk}})

    print(f"Upserting {len(vectors)} vectors")
    if vectors:
        print(f"Dimension of first vector: {len(vectors[0]['values'])}")
    else:
        print("No vectors to upsert!")
    index.upsert(vectors)


# Retrieve from Pinecone
def retrieve_relevant_documents(query, top_k=5):
    try:
        query_emb = openai_client.embeddings.create(
            model="text-embedding-3-small", input=query
        ).data[0].embedding

        results = index.query(vector=query_emb, top_k=top_k, include_metadata=True)
        return [match["metadata"]["text"] for match in results["matches"]]
    except Exception as e:
        logging.error(f"[ERROR] Retrieval failed: {e}")
        st.error("Document retrieval failed.")
        return []

# Answer generator using context
def generate_answer(query):
    docs = retrieve_relevant_documents(query, top_k=5)
    
    if docs:
        context = "\n\n".join(docs)
        prompt = f"""Use the following context to answer the query.
        
        ### Context:
        {context}
        
        ### Query:
        {query}
        
        ### Answer:"""
    else:
        prompt = f"""You are an intelligent AI assistant. Answer this:
        
        ### Query:
        {query}
        
        ### Answer:"""

    try:
        response = openai_client.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "You answer questions using provided document context. If no context, use your own knowledge."},
                {"role": "user", "content": prompt}
            ]
        )
        return response.choices[0].message.content
    except Exception as e:
        logging.error(f"[ERROR] GPT failed: {e}")
        st.error("Failed to generate answer.")
        return "Sorry, something went wrong."

# Streamlit UI
st.set_page_config(page_title="Document QA", page_icon="📄", layout="wide")
st.title(":bamboo: Bamboo Species Chatbot")
action = st.radio("Choose your action:", ("Upload New Files", "Ask a Question"))

if action == "Upload New Files":
    files = st.file_uploader("Upload PDF or DOCX files", type=["pdf", "docx"], accept_multiple_files=True)
    if files:
        for f in files:
            with st.spinner(f"Processing {f.name}..."):
                text = read_pdf(f) if f.name.endswith(".pdf") else read_docx(f)
                chunks = chunk_document(text)
                st.info("Generating embeddings...")
                embs = generate_embeddings(chunks)
                st.info("Storing to Pinecone...")
                store_in_pinecone(chunks, embs)
                st.success(f"{f.name} indexed to Pinecone!")

elif action == "Ask a Question":
    query = st.text_input("Ask something about your documents:", placeholder="Type here...")
    if st.button("Generate Answer"):
        if query.strip():
            with st.spinner("Thinking..."):
                answer = generate_answer(query)
                st.markdown("### 🤖 Answer:")
                st.write(answer)
        else:
            st.warning("Please enter a question.")
