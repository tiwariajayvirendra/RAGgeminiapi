import json
import faiss
import numpy as np

from sentence_transformers import SentenceTransformer

INDEX_FILE="data/faiss.index"
CHUNKS_FILE="data/chunks.json"

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

print("Loading Model...")
model = SentenceTransformer(MODEL_NAME)

print("Loding FAISS Index...")

index=faiss.read_index(INDEX_FILE)

with open (CHUNKS_FILE,"r",encoding='utf-8') as file:
    chunks=json.load(file)

def search(query,top_k=3):
    query_embedding=model.encode(
        [query],
        convert_to_numpy=True
        )
    query_embedding=query_embedding.astype("float32")
    
    distance, indices = index.search(
        query_embedding,
        top_k
    )
    
    
    results=[]
    
    for distance, index_id in zip(
        distance[0],
        indices[0]
        ):   
        #.......
        if index_id < 0:
            continue
        results.append(
            {"text":chunks[index_id],
             "distance": float(distance)
             })
    return results

if __name__=="__main__":
    question=input("Ask A question:")
    results=search(question)
    print("\nSearch Results:")
    print("*"*60)
    for i ,result in enumerate(results,start=1):
        print(f"\nResult {i}")
        print("-"*60)
        print(result['text'])
        print(
            "Distance:",
            result['distance']
            )


 