import json
from PyPDF2 import PdfReader

PDF_FILE = "documents/Ecommerce_C.pdf"
CHUNKS_FILE = "data/chunks.json"

def split_into_chunks(text, max_chars=500):
    lines = text.split("\n")
    chunks = []
    current_chunk = ""
    for line in lines:
        if len(current_chunk) + len(line) <= max_chars:
            current_chunk += line + " "
        else:
            chunks.append(current_chunk.strip())
            current_chunk = line + " "
    if current_chunk:
        chunks.append(current_chunk.strip())
    return chunks

def pdf_to_chunks(pdf_path):
    reader = PdfReader(pdf_path)
    all_chunks = []
    for page_num, page in enumerate(reader.pages):
        text = page.extract_text()
        if text:
            page_chunks = split_into_chunks(text)
            all_chunks.extend(page_chunks)
    return all_chunks

if __name__ == "__main__":
    chunks = pdf_to_chunks(PDF_FILE)
    with open(CHUNKS_FILE, "w", encoding="utf-8") as f:
        json.dump(chunks, f, ensure_ascii=False, indent=2)
    print(f"Extracted {len(chunks)} chunks from PDF")
