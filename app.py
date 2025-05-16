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
import pyrebase

# Load environment variables
load_dotenv()

# Firebase configuration
firebase_config = {
    "apiKey": "AIzaSyDdMC9baSTkWWBwnVCu-Xs2jZj86-fLmsE",
    "authDomain": "bamboo-project-d9832.firebaseapp.com",
    "databaseURL": "https://bamboo-project-d9832-default-rtdb.firebaseio.com",
    "projectId": "bamboo-project-d9832",
    "storageBucket": "bamboo-project-d9832.appspot.com",
    "messagingSenderId": "525252188687",
    "appId": "1:525252188687:web:366f42a40f6fe05678a17a",
    "measurementId": "G-1J9PH26HN0"
}

# Initialize Firebase
firebase = pyrebase.initialize_app(firebase_config)
auth = firebase.auth()

# Pinecone setup
PINECONE_API_KEY = os.environ.get("PINECONE_API_KEY")
pc = Pinecone(api_key=PINECONE_API_KEY)

# OpenAI setup
OPENAI_API_KEY = os.environ.get("CLIENT_KEY")
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

# Authentication functions
def handle_login(email, password):
    """Handle user login with Firebase."""
    try:
        user = auth.sign_in_with_email_and_password(email, password)
        st.session_state.user = user
        st.session_state.user_email = email
        st.success("Login successful!")
        st.rerun()
        return True
    except Exception as e:
        error_message = str(e)
        if "INVALID_EMAIL" in error_message:
            st.error("Invalid email address.")
        elif "INVALID_PASSWORD" in error_message:
            st.error("Incorrect password.")
        else:
            st.error(f"Login failed: {error_message}")
        return False

def handle_password_reset(email):
    """Send password reset email."""
    try:
        auth.send_password_reset_email(email)
        st.success(f"Password reset email sent to {email}")
    except Exception as e:
        error_message = str(e)
        if "INVALID_EMAIL" in error_message:
            st.error("This email is not registered.")
        else:
            st.error(f"Password reset failed: {error_message}")

def handle_logout():
    """Clear session state on logout."""
    for key in list(st.session_state.keys()):
        del st.session_state[key]
    st.success("You have been logged out successfully.")

def show_auth_page():
    """Display authentication options (login/forgot password)."""
    st.title(":bamboo: Bamboo Species Chatbot Login")
    st.caption("Access your account securely.")
        
    auth_option = st.radio("Choose an option:", 
                          ["Login", "Forgot Password"])
    
    if auth_option == "Login":
        with st.form("login_form"):
            email = st.text_input("Email", key="login_email")
            password = st.text_input("Password", type="password", key="login_password")
            submit = st.form_submit_button("Login")
            
            if submit:
                handle_login(email, password)
    
    elif auth_option == "Forgot Password":
        with st.form("reset_form"):
            email = st.text_input("Email", key="reset_email")
            submit = st.form_submit_button("Send Reset Link")
            
            if submit:
                handle_password_reset(email)

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

# Main app functionality
def show_main_app():
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

# Streamlit app
def main():
    st.set_page_config(page_title="Document QA", page_icon="📄", layout="wide")
    
    # Custom CSS for better styling
    st.markdown("""
    <style>
        .stTextInput input, .stTextArea textarea {
            border-radius: 8px !important;
        }
        .stButton button {
            border-radius: 8px !important;
            background-color: #4CAF50 !important;
            color: white !important;
        }
        .stAlert {
            border-radius: 8px !important;
        }
    </style>
    """, unsafe_allow_html=True)
    
    # Initialize session state for authentication
    if "user" not in st.session_state:
        st.session_state.user = None
    if "user_email" not in st.session_state:
        st.session_state.user_email = None
    
    # Show appropriate page based on authentication state
    if st.session_state.user:
        # User is logged in - show sidebar with logout button
        st.sidebar.title(f"Welcome, {st.session_state.user_email}")
        if st.sidebar.button("Logout"):
            handle_logout()
            st.rerun()
        
        # Show the main application
        show_main_app()
    else:
        # User is not logged in - show authentication page
        show_auth_page()

if __name__ == "__main__":
    main()