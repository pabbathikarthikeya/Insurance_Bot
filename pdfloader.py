import pdfplumber

def extract_text_from_pdf(pdf_path):
    """
    Extracts all text from a given PDF file.
    
    Args:
        pdf_path (str): Path to the PDF file.
        
    Returns:
        str: Full extracted text from the PDF.
    """
    full_text = ""
    
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            text = page.extract_text()
            if text:
                full_text += text + "\n"
    
    return full_text

if __name__ == "__main__":
    # Example usage
    pdf_path = "IRD.pdf"  # <-- change this to your file path
    extracted_text = extract_text_from_pdf(pdf_path)
    
    # Save extracted text to a file (optional)
    with open("extracted_text.txt", "w", encoding="utf-8") as f:
        f.write(extracted_text)
    
    print("✅ PDF extraction complete. Text saved to 'extracted_text.txt'.")
