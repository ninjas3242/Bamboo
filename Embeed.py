import os
import glob
import PyPDF2  # PyPDF2 for reading PDFs
import pinecone
from sentence_transformers import SentenceTransformer

# Initialize Pinecone client
from pinecone import Pinecone

# Setup Pinecone instance and index
pc = Pinecone(api_key="pcsk_5M4udd_38cet55MyU25Sbi59gA4ggxNXAfaZmVJtKjvWhVBJ9rsoiAHREVgJiP8xiw2FzA")  # Replace with your actual API key
index = pc.Index("bamboo")

# Model for embedding
model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

# Path to the folder containing pages (make sure this is correct)
folder_path = r"C:\Users\lenovo\Downloads\bambooooo files"  # Update with your actual folder path

# Function to extract text from PDF
def extract_text_from_pdf(pdf_path):
    text = ""
    with open(pdf_path, "rb") as f:
        reader = PyPDF2.PdfReader(f)
        for page in reader.pages:
            text += page.extract_text()  # Extract text from each page
    return text

# Recursively get all .pdf files in the folder and subdirectories
pdf_files = glob.glob(os.path.join(folder_path, "**/*.pdf"), recursive=True)

if not pdf_files:
    print("No PDF documents found in the folder.")
else:
    print(f"Found PDF files: {pdf_files}")
    for pdf_file in pdf_files:
        print(f"Processing PDF: {pdf_file}")
        # Extract text from the PDF
        text = extract_text_from_pdf(pdf_file)
        
        if text:  # Only process non-empty texts
            # Embed the extracted text
            embedding = model.encode(text)
            
            # Upsert the embedding into Pinecone
            try:
                index.upsert([(pdf_file, embedding)])
                print(f"Upserted {pdf_file} into Pinecone successfully.")
            except Exception as e:
                print(f"Error inserting {pdf_file} into Pinecone: {e}")
        else:
            print(f"No text found in {pdf_file}")
