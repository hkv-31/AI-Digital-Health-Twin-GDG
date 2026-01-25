# ocr_engine.py
import pdfplumber
from PIL import Image
import pytesseract

def extract_text(file) -> str:
    """
    Extract raw text from PDF or image uploaded via Streamlit.
    """
    text = ""

    if hasattr(file, "type") and file.type == "application/pdf":
        with pdfplumber.open(file) as pdf:
            for page in pdf.pages:
                text += (page.extract_text() or "") + "\n"
    else:
        image = Image.open(file)
        text = pytesseract.image_to_string(image)

    return text.strip()
