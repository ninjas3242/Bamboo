import os
import json
import requests
import streamlit as st
from dotenv import load_dotenv
from requests_aws4auth import AWS4Auth
from sample_openai import OpenAI
from langchain.text_splitter import RecursiveCharacterTextSplitter
from PyPDF2 import PdfReader
from docx import Document

# Load environment variables
load_dotenv()

# AWS credentials and region
AWS_ACCESS_KEY = os.getenv("AWS_ACCESS_KEY_ID")
AWS_SECRET_KEY = os.getenv("AWS_SECRET_ACCESS_KEY")
AWS_REGION = os.getenv("AWS_DEFAULT_REGION")

# OpenAI API Key
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
openai_client = OpenAI(api_key=OPENAI_API_KEY)

# OpenSearch Serverless endpoint
OPENSEARCH_ENDPOINT = "https://34b3vjfw9q910bdo9jx5.us-east-1.aoss.amazonaws.com"

# AWS Signature Version 4 authentication
awsauth = AWS4Auth(AWS_ACCESS_KEY, AWS_SECRET_KEY, AWS_REGION, "aoss")

# Define the index name
INDEX_NAME = "aoss-index"

# Document chunking settings
CHUNK_SIZE = 500
CHUNK_OVERLAP = 100

def read_pdf(file):
    pdf_reader = PdfReader(file)
    return "\n".join([page.extract_text() for page in pdf_reader.pages if page.extract_text()])

def read_docx(file):
    doc = Document(file)
    return "\n".join([para.text for para in doc.paragraphs])

def read_txt(file):
    return file.read().decode("utf-8")

def chunk_document(text):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP, length_function=len
    )
    return text_splitter.split_text(text)

def generate_embedding(text):
    response = openai_client.embeddings.create(input=text, model="text-embedding-ada-002")
    return response.data[0].embedding

def index_documents(chunks):
    bulk_data = []
    for chunk in chunks:
        embedding = generate_embedding(chunk)
        document = {"text": chunk, "osha_vector": embedding}
        bulk_data.append(json.dumps({"index": {"_index": INDEX_NAME}}))
        bulk_data.append(json.dumps(document))
    bulk_url = f"{OPENSEARCH_ENDPOINT}/_bulk"
    headers = {"Content-Type": "application/json"}
    response = requests.post(bulk_url, auth=awsauth, headers=headers, data="\n".join(bulk_data) + "\n")
    st.success("Documents indexed successfully!")

def process_document(uploaded_file):
    if uploaded_file is not None:
        if uploaded_file.name.endswith(".pdf"):
            text = read_pdf(uploaded_file)
        elif uploaded_file.name.endswith(".docx"):
            text = read_docx(uploaded_file)
        elif uploaded_file.name.endswith(".txt"):
            text = read_txt(uploaded_file)
        else:
            st.error("Unsupported file type.")
            return
        chunks = chunk_document(text)
        index_documents(chunks)

def generate_query_embedding(query):
    response = openai_client.embeddings.create(input=query, model="text-embedding-ada-002")
    return response.data[0].embedding

def search_documents(query, k=5):
    query_embedding = generate_query_embedding(query)
    search_body = {
        "size": k,
        "_source": ["text"],
        "query": {"knn": {"osha_vector": {"vector": query_embedding, "k": k}}}
    }
    search_url = f"{OPENSEARCH_ENDPOINT}/{INDEX_NAME}/_search"
    headers = {"Content-Type": "application/json"}
    response = requests.get(search_url, auth=awsauth, headers=headers, data=json.dumps(search_body))
    if response.status_code == 200:
        results = response.json().get("hits", {}).get("hits", [])
        return [hit["_source"]["text"] for hit in results]
    return []

st.title("Streamlit RAG with OpenSearch")
uploaded_file = st.file_uploader("Upload a document", type=["pdf", "docx", "txt"])
if uploaded_file:
    process_document(uploaded_file)
query = st.text_input("Enter search query")
if st.button("Search"):
    results = search_documents(query)
    st.write("## Search Results")
    for res in results:
        st.write(res)
