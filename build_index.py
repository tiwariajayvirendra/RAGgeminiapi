import faiss
import numpy as np
import json
from sentence_transformers import SentenceTransformer

CHUNKS_FILE = "data/chunks.json"
INDEX_FILE = "data/faiss.index"
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

def build_index():
    model = SentenceTransformer(MODEL_NAME)
    with open(CHUNKS_FILE, "r", encoding="utf-8") as f:
        chunks = json.load(f)

    embeddings = model.encode(chunks, convert_to_numpy=True).astype("float32")
    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(embeddings)

    faiss.write_index(index, INDEX_FILE)
    print("FAISS index built and saved.")

if __name__ == "__main__":
    build_index()
