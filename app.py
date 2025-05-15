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
import ollama  # <-- Local embedding model
from openai import OpenAI

# Load environment variables
load_dotenv()

# Pinecone setup
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
pc = Pinecone(api_key=PINECONE_API_KEY)

# OpenAI API Key
CLIENT_KEY = os.getenv("CLIENT_KEY")
openai_client = OpenAI(api_key=CLIENT_KEY)

INDEX_NAME = "bamboo"
DIMENSION = 384  # Using 384-dim embeddings
REGION = "us-east-1"

# Create index if not already created
if INDEX_NAME not in pc.list_indexes().names():
    pc.create_index(
        name=INDEX_NAME,
        dimension=DIMENSION,
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", region=REGION)
    )

# Access the index
index = pc.Index(INDEX_NAME)

# Document chunking settings
CHUNK_SIZE = 2000
CHUNK_OVERLAP = 500

# Configure logging
logging.basicConfig(level=logging.INFO)

# PDF Reader
def read_pdf(file):
    file.seek(0)
    pdf_reader = PdfReader(BytesIO(file.read()))
    text = []
    for page in pdf_reader.pages:
        page_text = page.extract_text()
        if page_text:
            text.append(page_text)
    return "\n".join(text) if text else "Could not extract text from PDF."

# DOCX Reader
def read_docx(file):
    file.seek(0)
    doc = Document(BytesIO(file.read()))
    return "\n".join([para.text for para in doc.paragraphs if para.text.strip()])

# Chunk text
def chunk_document(text):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        length_function=len,
        separators=["\n\n", "\n", " ", ""]
    )
    return text_splitter.split_text(text)

# Generate embeddings using Ollama (local, free)
def generate_embeddings(text_chunks):
    embeddings = []
    for i, chunk in enumerate(text_chunks):
        try:
            # Get the response from Ollama
            response = ollama.embed(model="mxbai-embed-large", input=chunk)

            # Check if 'embeddings' is in the response and is a list
            if 'embeddings' in response and isinstance(response['embeddings'], list):
                # Check if there are embeddings in the list
                if len(response['embeddings']) > 0:
                    embedding = response['embeddings'][0]  # Access the first embedding
                    embeddings.append(embedding)
                else:
                    raise ValueError(f"Empty embeddings list for chunk {i}")
            else:
                raise ValueError(f"Embeddings not found in response for chunk {i}: {response}")

        except Exception as e:
            logging.error(f"[ERROR] Failed to generate embedding for chunk {i}: {e}")
            st.error(f"Embedding generation failed for chunk {i}. See terminal for details.")
    
    return embeddings

# Store chunks + embeddings in Pinecone
def store_in_pinecone(chunks, embeddings):
    vectors = []
    for chunk, embedding in zip(chunks, embeddings):
        vector_id = str(uuid4())
        vectors.append({
            "id": vector_id,
            "values": embedding,
            "metadata": {"text": chunk}
        })
    index.upsert(vectors)

# Search Pinecone
def retrieve_relevant_documents(query, top_k=5):
    try:
        response = ollama.embed(
            model="mxbai-embed-large",
            input=query
        )

        # Check if 'embedding' is in the response
        if 'embeddings' in response:
            query_embedding = response['embeddings']
        else:
            # Print the full response for debugging
            logging.error(f"Embedding not found in response: {response}")
            st.error("Error: Embedding not found in response. See terminal for details.")
            return []

        results = index.query(vector=query_embedding, top_k=top_k, include_metadata=True)
        return [match["metadata"]["text"] for match in results["matches"]]
    except Exception as e:
        logging.error(f"[ERROR] Failed to retrieve relevant documents: {e}")
        st.error("Error during document retrieval. See terminal for details.")
        return []

# Function to generate an answer using retrieved document context
def generate_answer(query):
    retrieved_docs = retrieve_relevant_documents(query, top_k=5)
    
    if retrieved_docs and any(retrieved_docs):
        context = "\n\n".join(retrieved_docs[:5])
        prompt = f"""Use the provided document context to answer the query.
        
        ### Document Context:
        {context}
        
        ### Query:
        {query}
        
        ### Answer:
        """
    else:
        prompt = f"""You are an expert AI assistant. Answer the following query using your general knowledge:
        
        ### Query:
        {query}
        
        ### Answer:
        """
    
    try:
        response = openai_client.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "You are an AI assistant that answers queries based on document context. If no documents are available, use general knowledge."},
                {"role": "user", "content": prompt}
            ]
        )
        
        return response.choices[0].message.content
    except Exception as e:
        logging.error(f"[ERROR] Failed to generate answer: {e}")
        st.error("Error during answer generation. See terminal for details.")
        return "I'm sorry, I couldn't generate an answer at this time. Please try again later."

# Streamlit UI
st.set_page_config(page_title="Document QA", page_icon="📄", layout="wide")
st.title(":bamboo: Bamboo Species Chatbot")
user_choice = st.radio("Choose your action:", ("Upload New Files", "Ask a Question"))

if user_choice == "Upload New Files":
    uploaded_files = st.file_uploader("Upload PDFs or DOCX files", accept_multiple_files=True, type=["pdf", "docx"])
    
    if uploaded_files:
        for uploaded_file in uploaded_files:
            with st.spinner(f"Processing {uploaded_file.name}..."):
                if uploaded_file.name.endswith(".pdf"):
                    text = read_pdf(uploaded_file)
                elif uploaded_file.name.endswith(".docx"):
                    text = read_docx(uploaded_file)
                else:
                    st.error(f"Unsupported file type: {uploaded_file.name}")
                    continue

                st.info("Chunking document and generating embeddings...")
                chunks = chunk_document(text)
                embeddings = generate_embeddings(chunks)

                st.info(f"Storing {len(chunks)} chunks to Pinecone...")
                store_in_pinecone(chunks, embeddings)

                st.success(f"Indexed {uploaded_file.name} to Pinecone!")

elif user_choice == "Ask a Question":
    query = st.text_input("Ask a question about the documents:", placeholder="Enter your query here...")

    if st.button("Generate Answer"):
        if query.strip():
            with st.spinner("Generating answer..."):
          
                
                # Then, generate and display the answer
                answer = generate_answer(query)
                st.markdown("### 🤖 Answer:")
                st.write(answer)
        else:
            st.warning("Please enter a query before clicking the button.")