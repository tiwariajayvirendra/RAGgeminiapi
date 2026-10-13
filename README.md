# RAG Gemini API 
This project implements a **Retrieval-Augmented Generation (RAG)** pipeline using:
- **FAISS** for vector search
- **Sentence-Transformers** for embeddings
- **Google Gemini API** for answer generation
- **Streamlit** for interactive UI

It allows you to upload large documents (like PDFs), split them into chunks, build a FAISS index, and then query them with natural language questions.

---

## 📂 Project Structure

RagPipe/

documents/              # Raw source documents (e.g., Ecommerce_C.pdf)
 

data/                   # Processed data (chunks.json, faiss.index)


 app.py                  # Streamlit UI
 

rag.py                  # RAG pipeline logic
 

search.py               # FAISS search functions
 

build_chunks.py         # Extract chunks from PDF
 

build_index.py          # Build FAISS index from chunks
 

requirements.txt        # Dependencies 

.env                    # Environment variables (API keys)