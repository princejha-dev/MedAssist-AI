import os
from pathlib import Path
from typing import List, Any

from dotenv import load_dotenv

from backend.rag.data_retrieval import load_retriever

# import whichever LLM you intend to use; we default to Gemini/Google
from langchain_google_genai import ChatGoogleGenerativeAI
# fallback example (uncomment if using OpenAI):
# from langchain_openai import ChatOpenAI


class MedicalChatbot:
    """Small wrapper around retrieval + LLM to answer medical questions.

    The object loads configuration from environment variables and keeps the
    retriever and model instances cached for reuse.  It exposes a single
    ``ask(question)`` method that returns text.
    """

    SYSTEM_INSTRUCTIONS = (
        """
You are a professional medical knowledge assistant.
You will be given context from trusted medical documents.  You must answer
questions using ONLY that context.  If the information is not contained in
context, respond with "I don't have enough medical information".

Safety guidelines:
* Do NOT diagnose, prescribe, suggest dosages, or provide treatment plans.
* Do NOT give legal, financial or medical advice.
* Always encourage users to consult a qualified healthcare professional.
* Do NOT hallucinate details or make up sources.
* Keep the tone neutral, factual, and simple.

Answer in clear language; use bullet points where appropriate.
""".strip()
    )

    def __init__(self):
        load_dotenv()  # read .env if present

        # choose an LLM implementation based on environment
        self.llm = self._init_llm()

        # keep a retriever loaded once
        self.retriever = load_retriever()

    def _init_llm(self):
        # currently supporting Google Gemini or OpenAI (as examples)
        gemini_key = os.getenv("GEMINI_API_KEY")
        if gemini_key:
            os.environ["GEMINI_API_KEY"] = gemini_key
            return ChatGoogleGenerativeAI(model="gemini-2.5-flash-lite")

        openai_key = os.getenv("OPENAI_API_KEY")
        if openai_key:
            os.environ["OPENAI_API_KEY"] = openai_key
            # import here to avoid a hard dependency if OpenAI is not installed
            try:
                from langchain_openai import ChatOpenAI
            except ImportError:
                raise RuntimeError("OPENAI_API_KEY set but langchain-openai package is missing")
            return ChatOpenAI(model_name="gpt-3.5-turbo")

        raise RuntimeError("No supported LLM API key found in environment.")

    def _build_prompt(self, question: str, contexts: List[str]) -> str:
        joined = "\n\n".join(contexts)
        prompt = (
            f"{self.SYSTEM_INSTRUCTIONS}\n\n"
            f"Context:\n{joined}\n\n"
            f"User question:\n{question}\n\n"
            "Answer:"
        )
        return prompt

    def ask(self, question: str) -> str:
        """Answer a single free‑text question."""

        if not question or not question.strip():
            return "Please provide a question."

        docs = self.retriever.get_relevant_documents(question)
        contexts = [d.page_content for d in docs if d.page_content]
        prompt = self._build_prompt(question, contexts)

        # call the LLM; return raw text
        response = self.llm.invoke(prompt)
        # the ``invoke`` method returns a ChatResult-like object with `.content`
        return response.content.strip()


# export a singleton for simple imports
assistant = MedicalChatbot()

