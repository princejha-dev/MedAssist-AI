# importing the required libraries

from pathlib import Path
from typing import Tuple

from langchain_community.document_loaders import PyPDFLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS


def ingest_documents(
    pdf_path: Path | str,
    index_folder: Path | str = "../vectorDB",
    index_name: str = "Faiss_index",
    chunk_size: int = 800,
    chunk_overlap: int = 150,
    embedding_model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
) -> Tuple[FAISS, HuggingFaceEmbeddings]:
    """Load a PDF, split it into chunks, create embeddings and store in a FAISS index.

    Arguments:
        pdf_path: location of the PDF file to ingest.
        index_folder: directory where the FAISS index will be saved.
        index_name: name for the index files.
        chunk_size: maximum tokens per chunk.
        chunk_overlap: overlap between chunks.
        embedding_model_name: HuggingFace model for embeddings.

    Returns:
        A tuple of (vector_db, embedding_model) for further use.
    """

    pdf_path = Path(pdf_path)
    if not pdf_path.is_file():
        raise FileNotFoundError(f"PDF not found at {pdf_path}")

    # load documents
    loader = PyPDFLoader(file_path=str(pdf_path))
    documents = loader.load()

    # split into chunks
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )
    chunks = splitter.split_documents(documents)

    # create embedding model
    embedding_model = HuggingFaceEmbeddings(model_name=embedding_model_name)

    # build vector store
    vector_db = FAISS.from_documents(documents=chunks, embedding=embedding_model)

    # save to disk
    index_folder = Path(index_folder)
    index_folder.mkdir(parents=True, exist_ok=True)
    vector_db.save_local(folder_path=str(index_folder), index_name=index_name)

    return vector_db, embedding_model


if __name__ == "__main__":
    # simple command‑line utility to rebuild the index
    import argparse

    parser = argparse.ArgumentParser(description="Ingest PDF into FAISS index")
    parser.add_argument("pdf", help="Path to the PDF file")
    parser.add_argument("--output", default="../vectorDB", help="Index directory")
    args = parser.parse_args()

    ingest_documents(pdf_path=args.pdf, index_folder=args.output)
    print("Ingestion complete. Index saved to", args.output)
