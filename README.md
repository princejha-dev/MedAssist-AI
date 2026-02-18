# Medical Assistance Chatbot (RAG)

This repository contains a retrieval‑augmented generation (RAG) medical assistant.
The backend uses **FastAPI** to serve a simple chat API; documents are indexed
with LangChain and FAISS.  The frontend is a React single–page app that
allows users to ask medical questions and receive answers generated from the
indexed material.

---

## Workspace structure

```
backend/           # Python FastAPI service + RAG logic
  rag/             # ingestion & retrieval modules
frontend/          # React chat user interface
documents/         # source PDF(s) used for indexing
vectorDB/          # generated FAISS index
notebook/          # exploratory Jupyter notebook (demo)
```

## Getting started

### Python backend

1. **Create a virtual environment** and install dependencies:
   ```bash
   python -m venv .venv
   # Windows
   .\.venv\Scripts\activate
   # *nix
   # source .venv/bin/activate

   pip install -r requirements.txt
   pip install fastapi uvicorn python-dotenv
   ```

2. **Prepare vector store** (only once or when documents change):
   ```bash
   python -m backend.rag.data_ingestion documents/medical_book.pdf
   ```
   This will create `vectorDB/Faiss_index` that the server uses at runtime.

3. **Set API key(s)** in a `.env` file at the project root:
   ```properties
   # choose one provider
   GEMINI_API_KEY=your_google_gemini_key
   # or
   OPENAI_API_KEY=your_openai_key
   ```

4. **Run the backend**:
   ```bash
   uvicorn backend.main:app --reload
   ```
   The service listens on `http://localhost:8000`.

### React frontend

1. Change into the frontend directory:
   ```bash
   cd frontend
   npm install
   npm start
   ```
2. Open `http://localhost:3000` in your browser and start chatting. The UI
   will send POST requests to the FastAPI server to obtain answers.

---

## Development notes

* The prompt template is defined in `backend/model.py` and includes strict
  safety guidelines and usage instructions to minimize hallucinations.
* Retrieval logic is cached and reuses the embedding model for efficiency.
* CORS is enabled for localhost origins so the React client can communicate
  with the API during development.
* You can extend `backend/main.py` with authentication, logging, or additional
  endpoints as needed.

---

Feel free to enhance styles, swap the LLM provider, or add more medical
documents to the corpus.
