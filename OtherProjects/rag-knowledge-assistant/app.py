import streamlit as st
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
from langchain.llms import HuggingFacePipeline
from transformers import pipeline

st.title("📚 RAG Knowledge Assistant")

uploaded_file = st.file_uploader("Upload a PDF", type="pdf")

if uploaded_file:
    # 1. Load PDF
    loader = PyPDFLoader(uploaded_file.name)
    docs = loader.load()

    # 2. Split text into chunks
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = splitter.split_documents(docs)

    # 3. Create embeddings
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    # 4. Store embeddings in FAISS
    db = FAISS.from_documents(chunks, embeddings)

    # 5. Load a HuggingFace model locally (small for testing)
    model_name = "google/flan-t5-base"  # you can swap with bigger models if GPU available
    pipe = pipeline("text2text-generation", model=model_name)
    local_llm = HuggingFacePipeline(pipeline=pipe)

    # 6. Create RetrievalQA
    qa = RetrievalQA.from_chain_type(
        llm=local_llm,
        retriever=db.as_retriever(search_kwargs={"k": 3})
    )

    # 7. User query
    query = st.text_input("Ask a question:")
    if query:
        answer = qa.run(query)
        st.write("### Answer:")
        st.write(answer)
