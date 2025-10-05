import os
import streamlit as st
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np

st.title("🔍 Semantic Search Engine")

# --- Load corpus ---
corpus_folder = "corpus"
documents = []
filenames = []

for file in os.listdir(corpus_folder):
    if file.endswith(".txt"):
        with open(os.path.join(corpus_folder, file), "r", encoding="utf-8") as f:
            text = f.read()
            documents.append(text)
            filenames.append(file)

# --- Create embeddings ---
@st.cache_resource  # caches the model for faster reloads
def load_model():
    return SentenceTransformer('all-MiniLM-L6-v2')

model = load_model()

@st.cache_resource
def build_faiss_index(docs):
    embeddings = model.encode(docs, convert_to_numpy=True)
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings)
    return index, embeddings

index, embeddings = build_faiss_index(documents)

# --- Query input ---
query = st.text_input("Enter your search query:")

if query:
    query_embedding = model.encode([query], convert_to_numpy=True)
    k = st.slider("Number of results", 1, 10, 3)
    distances, indices = index.search(query_embedding, k)

    st.subheader("Top Results:")
    for rank, idx in enumerate(indices[0]):
        st.markdown(f"**{rank+1}. {filenames[idx]}** (distance={distances[0][rank]:.4f})")
        st.write(documents[idx][:500] + "...")
