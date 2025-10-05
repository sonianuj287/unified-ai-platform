import os
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.llms import OpenAI
from langchain.chains import RetrievalQA


# 1. Load document
loader = PyPDFLoader("sample.pdf")
docs = loader.load()

# 2. Split into chunks
splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
chunks = splitter.split_documents(docs)

# 3. Create embeddings
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# 4. Store in FAISS
db = FAISS.from_documents(chunks, embeddings)

# 5. Build QA chain
qa = RetrievalQA.from_chain_type(
    llm=OpenAI(temperature=0), 
    retriever=db.as_retriever(search_kwargs={"k": 3})
)

# 6. Ask questions
while True:
    query = input("Ask a question: ")
    if query.lower() in ["exit", "quit"]:
        break
    print("Answer:", qa.run(query))