So the steps are:

Load documents (PDFs or text).

Create embeddings for each chunk.

Store in FAISS (vector DB).

Take user query, convert to embedding.

Retrieve most relevant chunks.

Feed chunks + query → LLM → get final answer.