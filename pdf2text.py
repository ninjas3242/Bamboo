import pdfplumber
import pytesseract
from pdf2image import convert_from_path
from langchain.text_splitter import RecursiveCharacterTextSplitter
import boto3

# Set Tesseract path for Windows (modify if needed)
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

# Amazon Translate character limit
MAX_TRANSLATE_LENGTH = 10000  

# Supported OCR languages for Tesseract
OCR_LANGUAGES = {
    "en": "eng",        
    "zh-cn": "chi_sim",  
    "zh-tw": "chi_tra",  
    "ja": "jpn",        
    "ko": "kor",
    "es": "spa",        # Spanish
    "fr": "fra",        # French
    "pt": "por"         # Portuguese
}

def detect_language(text):
    """Detects the dominant language using Amazon Comprehend."""
    try:
        comprehend = boto3.client("comprehend")
        response = comprehend.detect_dominant_language(Text=text)
        languages = response.get("Languages", [])
        return languages[0]["LanguageCode"] if languages else "auto"
    except Exception as e:
        print(f"Language detection error: {e}")
        return "auto"

def translate_text(text, source_lang, target_lang="en"):
    """Translates text using Amazon Translate."""
    try:
        translate = boto3.client("translate")
        response = translate.translate_text(
            Text=text,
            SourceLanguageCode=source_lang,
            TargetLanguageCode=target_lang
        )
        return response["TranslatedText"]
    except Exception as e:
        print(f"Translation error: {e}")
        return text  # Return original text if translation fails

def split_text_with_recursive_splitter(text, chunk_size=1000, chunk_overlap=50):
    """Splits text into manageable chunks for translation."""
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    return splitter.split_text(text)

def extract_text_from_pdf(pdf_path):
    """Extracts text from a scanned PDF using OCR with auto-detected language."""
    extracted_text = ""
    
    # Convert PDF to images
    images = convert_from_path(pdf_path)

    for img in images:
        # Try default OCR (English)
        ocr_text = pytesseract.image_to_string(img, lang="eng").strip()
        
        # If English OCR fails, detect and apply another language
        if not ocr_text:
            detected_lang = detect_language(ocr_text)
            print(f"Detected Language: {detected_lang}")
            
            ocr_lang = OCR_LANGUAGES.get(detected_lang, "eng")  # Default to English
            print(f"Using OCR language: {ocr_lang}")
            
            ocr_text = pytesseract.image_to_string(img, lang=ocr_lang).strip()
        
        extracted_text += ocr_text + "\n"

    return extracted_text.strip()

def pdf_to_text(pdf_path, txt_path, translate=False):
    """Extracts, translates (if needed), and saves text from a PDF."""
    try:
        # Extract text from PDF (OCR)
        text = extract_text_from_pdf(pdf_path)

        if not text:
            print("No text found, even after OCR.")
            return

        # Split text into chunks
        text_chunks = split_text_with_recursive_splitter(text)

        translated_chunks = []
        if translate:
            source_lang = detect_language(text)
            print(f"Detected Source Language: {source_lang}")

            if source_lang != "en":
                for chunk in text_chunks:
                    translated_chunks.append(translate_text(chunk, source_lang, target_lang="en"))
            else:
                translated_chunks = text_chunks
        else:
            translated_chunks = text_chunks

        final_text = "\n".join(translated_chunks)

        with open(txt_path, "w", encoding="utf-8") as txt_file:
            txt_file.write(final_text)

        print(f"Text extracted and saved to {txt_path}")
    except Exception as e:
        print(f"Error: {e}")

# Example usage
pdf_to_text(r"D:\Bamboo Chatbot\Data Ingestion\Page7\2025-01-08-BRU_5-3_中文-2稿.pdf", "output.txt", translate=True)
