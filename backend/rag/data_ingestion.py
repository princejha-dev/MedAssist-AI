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
