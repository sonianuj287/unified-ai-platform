import os
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np

# 1. Load corpus
corpus_folder = "corpus"
documents = []
filenames = []

for file in os.listdir(corpus_folder):
    if file.endswith(".txt"):
        with open(os.path.join(corpus_folder, file), "r", encoding="utf-8") as f:
            text = f.read()
            documents.append(text)
            filenames.append(file)

# 2. Create embeddings
model = SentenceTransformer('all-MiniLM-L6-v2')
embeddings = model.encode(documents, convert_to_numpy=True)

# 3. Build FAISS index
dimension = embeddings.shape[1]
index = faiss.IndexFlatL2(dimension)  # L2 similarity
index.add(embeddings)

print(f"Indexed {len(documents)} documents.")

# 4. Query loop
while True:
    query = input("Enter your search query (or 'exit'): ")
    if query.lower() == "exit":
        break

    query_embedding = model.encode([query], convert_to_numpy=True)
    k = 3  # top-3 results
    distances, indices = index.search(query_embedding, k)

    print("\nTop results:")
    for rank, idx in enumerate(indices[0]):
        print(f"{rank+1}. {filenames[idx]} (distance={distances[0][rank]:.4f})")
        print(documents[idx][:300], "...\n")  # show first 300 chars
