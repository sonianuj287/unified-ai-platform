import os
import streamlit as st
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA

# Import Gemini LLM from LangChain Google GenAI provider
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage

# Replace with your actual key
os.environ["GEMINI_API_KEY"] = ""

# Set your Gemini API key
# Option A: environment variables
if "GOOGLE_API_KEY" not in os.environ:
    # optionally prompt or fallback
    st.warning("You need to set the GOOGLE_API_KEY environment variable")
    # You might also put a text_input here to get key from user
else:
    pass  # okay

st.title("📚 RAG Knowledge Assistant (with Gemini API)")

uploaded_file = st.file_uploader("Upload a PDF", type="pdf")

if uploaded_file:
    loader = PyPDFLoader(uploaded_file.name)
    docs = loader.load()

    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = splitter.split_documents(docs)

    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    db = FAISS.from_documents(chunks, embeddings)

    # Initialize Gemini LLM
    gemini_llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",   # change this to the model you prefer
        temperature=0,
        google_api_key=os.environ.get("GOOGLE_API_KEY"),
        # you can add other config params if needed
    )

    # Use LangChain’s RetrievalQA chain with this LLM
    qa = RetrievalQA.from_chain_type(
        llm=gemini_llm,
        retriever=db.as_retriever(search_kwargs={"k": 3})
    )

    query = st.text_input("Ask a question:")
    if query:
        with st.spinner("Thinking..."):
            # Create a HumanMessage for Gemini
            # Note: For simple use, you can directly invoke with string; but Gemini API via langchain often expects message structure
            # We'll call via the chain
            answer = qa.run(query)
        st.write("### Answer:")
        st.write(answer)
