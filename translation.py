import streamlit as st
import boto3
import hashlib
from io import BytesIO
import PyPDF2
from docx import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from concurrent.futures import ThreadPoolExecutor

# Function to extract text from PDF file
def extract_text_from_pdf(pdf_file):
    pdf_reader = PyPDF2.PdfReader(BytesIO(pdf_file))
    text = ""
    for page in pdf_reader.pages:
        text += page.extract_text() or ""  # Handle None cases
    return text

# Function to extract text from DOCX file
def extract_text_from_docx(docx_file):
    doc = Document(BytesIO(docx_file))
    return "\n".join(para.text for para in doc.paragraphs)

# Function to translate a text chunk using Amazon Translate
def translate_text_chunk(text_chunk, target_language="en"):
    translate = boto3.client("translate")
    response = translate.translate_text(
        Text=text_chunk,
        SourceLanguageCode="auto",  
        TargetLanguageCode=target_language
    )
    return response["TranslatedText"]

# Function to translate a document (split into chunks)
def translate_document(text, target_language="en"):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=4000, chunk_overlap=100)
    chunks = text_splitter.split_text(text)
    translated_chunks = [None] * len(chunks)
    
    progress_bar = st.progress(0)  # Initialize progress bar
    
    def process_chunk(i, chunk):
        return i, translate_text_chunk(chunk, target_language)

    with ThreadPoolExecutor() as executor:
        for i, translated_chunk in executor.map(lambda x: process_chunk(*x), enumerate(chunks)):
            translated_chunks[i] = translated_chunk
            progress_bar.progress((i + 1) / len(chunks))  # Update progress bar in the main thread
    
    return ''.join(translated_chunks)

# Streamlit UI
st.title("Amazon Translate for PDF and Word Files")
files = st.file_uploader("Upload PDF or Word files", type=["pdf", "docx"], accept_multiple_files=True)

if files:
    processed_files = set()
    all_extracted_texts = []
    all_translated_texts = []
    
    for file in files:
        file_content = file.read()
        file_hash = hashlib.md5(file_content).hexdigest()
        
        if file_hash in processed_files:
            st.warning(f"Duplicate file detected: {file.name}. Skipping.")
            continue
        
        processed_files.add(file_hash)
        st.write(f"Processing: {file.name}")

        # Extract text based on file type
        if file.type == "application/pdf":
            extracted_text = extract_text_from_pdf(file_content)
        elif file.type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
            extracted_text = extract_text_from_docx(file_content)

        if not extracted_text.strip():
            st.warning(f"No text found in {file.name}.")
            continue

        all_extracted_texts.append(extracted_text)
        translated_text = translate_document(extracted_text)
        all_translated_texts.append(translated_text)

    if all_translated_texts:
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Original Texts")
            st.text_area("Original", '\n\n'.join(all_extracted_texts), height=300)
        with col2:
            st.subheader("Translated Texts")
            st.text_area("Translated", '\n\n'.join(all_translated_texts), height=300)
        st.success("Translation complete!")

    # Start chunking for embedding
    st.write("Starting chunking for embedding...")
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = text_splitter.split_text('\n\n'.join(all_translated_texts))
    st.write(f"Total chunks created: {len(chunks)}")
