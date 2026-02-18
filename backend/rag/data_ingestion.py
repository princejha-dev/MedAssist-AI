# importing the required libraries

from langchain_community.document_loaders import PyPDFLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
import os


#loading the document
document=PyPDFLoader(
    file_path="..\documents\medical_book.pdf"
)

document=document.load()


#splitting the document into chunk
splitter = RecursiveCharacterTextSplitter(
    chunk_size=800,
    chunk_overlap=150
)

chunks = splitter.split_documents(document)


#embedding the document(chunks)

embedding_model=HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

vector_db=FAISS.from_documents(
    documents=chunks,
    embedding=embedding_model
)

vector_db.save_local(folder_path="../vectorDB",index_name="Faiss_index")