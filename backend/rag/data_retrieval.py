from pathlib import Path
from typing import Optional, Any

from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS


def load_retriever(
    index_folder: Path | str = "../vectorDB",
    index_name: str = "Faiss_index",
    embedding_model: Optional[HuggingFaceEmbeddings] = None,
    k: int = 6,
) -> Any:
    """Load a FAISS index from disk and return a retriever object.

    If `embedding_model` is not provided, a new one will be created with the
    default miniLM model. Loading a fresh model each time is cheap since it is
    just the embedding class, not the heavy transformer weights.
    """

    index_folder = Path(index_folder)
    if not index_folder.exists():
        raise FileNotFoundError(f"Index folder does not exist: {index_folder}")

    if embedding_model is None:
        embedding_model = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )

    vector_db = FAISS.load_local(
        folder_path=str(index_folder),
        index_name=index_name,
        embeddings=embedding_model,
        allow_dangerous_deserialization=True,
    )

    retriever = vector_db.as_retriever(search_kwargs={"k": k})
    return retriever


# convenience instance exported for backwards compatibility
try:
    retriever = load_retriever()
except Exception:  # pragma: no cover
    retriever = None
