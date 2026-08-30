#load pdf
#split into chunks
#create embeddings
#store in chroma db
import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_mistralai import MistralAIEmbeddings
# from langchain_huggingface import HuggingFaceEmbeddings
from dotenv import load_dotenv
load_dotenv()


data = PyPDFLoader("document loaders/fundamentals-of-deep-learning-designing-next-generation-machine-intelligence-algorithms-first-edition-9781491925614-1491925612_compress.pdf")
docs=data.load()

splitter=RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=20)
chunks=split_docs=splitter.split_documents(docs)

for chunk in chunks:
    chunk.page_content = chunk.page_content.encode(
        "ascii", errors="ignore"
    ).decode("ascii")

embedding_model=MistralAIEmbeddings(
    model="mistral-embed")

# embedding_model=HuggingFaceEmbeddings(
#     model_name="sentence-transformers/all-MiniLM-L6-v2"
# )



vectorstore=Chroma.from_documents(
    documents=chunks,
    embedding=embedding_model,
    persist_directory="chroma_db"
)
print("✅ Vector database created successfully!")