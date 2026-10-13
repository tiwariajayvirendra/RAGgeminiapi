import os
import json
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

# Reading the File Documents / company.txt

DOCUMENT_FILE="documents/company.txt"

# reading another directory that is DATA folder 
DATA_DIR = "data"

# Chunking the file of 
CHUNKS_FILE = os.path.join(DATA_DIR,"chunks.json")

INDEX_FILE = os.path.join(DATA_DIR,"faiss.index")

CHUNK_SIZE=1000
CHUNK_OVERLAP = 100

# Model Using
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

def load_documents():
    with open(DOCUMENT_FILE,'r',encoding='utf-8') as  f:
        text= f.read()

        return text

def create_chunks(text):
    chunks=[]
    start=0
    while start < len(text):
        end=start+CHUNK_SIZE
        chunk=text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        start += CHUNK_SIZE-CHUNK_OVERLAP
    return chunks


#Embedding taiyar karna. 

def create_embeddings(chunks):
    print("Loading Embedding Model...")
    model=SentenceTransformer(MODEL_NAME)
    embeddings=model.encode(chunks,show_progress_bar=True,convert_to_numpy=True)
    return embeddings

def create_faiss_index(embeddings):
    embeddings=embeddings.astype("float32")
    dimension=embeddings.shape[1]
    print("Embedding Dimention", dimension)
    index=faiss.IndexFlatL2(dimension)
    index.add(embeddings)
    return index

def save_data(chunks,index):
    os.makedirs(DATA_DIR,exist_ok=True)
    with open(CHUNKS_FILE,"w",encoding="utf-8") as f:
        json.dump(chunks, f, ensure_ascii=False, indent=4)
    faiss.write_index(index,INDEX_FILE)
    print("FAISS Index Saved",INDEX_FILE)

def main():
    print("Loading Documents...")
    text=load_documents()
    print("Document Characters:",len(text))
    print("Creating Chunks..")
    chunks=create_chunks(text)
    print("Number of Chunks:",len(chunks))
    print("Creating Embeddings...")
    embeddings=create_embeddings(chunks)
    print("Creating FAISS Index...")
    index=create_faiss_index(embeddings)
    print("Saving Data...")
    save_data(chunks,index)
    print("Ingestion Completed Sucessfully")

if __name__=="__main__":
    main()



