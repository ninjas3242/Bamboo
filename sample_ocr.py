import pytesseract
from pdf2image import convert_from_path
from PIL import Image
import os

# Set Tesseract path if needed (Windows users)
# pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

def extract_text_from_pdf(pdf_path):
    images = convert_from_path(pdf_path)  # Convert PDF to images
    extracted_text = ""

    for i, img in enumerate(images):
        text = pytesseract.image_to_string(img)  # Extract text from each page
        extracted_text += f"\n--- Page {i+1} ---\n{text}"

    return extracted_text

# Example usage
pdf_file = r"D:\Bamboo Chatbot\Data Ingestion\Page7\2025-01-08-BRU_5-3_中文-2稿.pdf"  # Replace with your PDF file
text_output = extract_text_from_pdf(pdf_file)

# Save to a text file
with open("output.txt", "w", encoding="utf-8") as f:
    f.write(text_output)

print("Text extraction complete. Check output.txt")
