"""
RAG layer for stable, slow-changing travel knowledge (visa rules, best time to
visit, local customs, packing guides) - things that don't need a live API
call every time and where you want the agent grounded in curated text instead
of a fresh LLM hallucination.

Two backends are supported:

1. "tfidf" (default): scikit-learn TF-IDF + cosine similarity, persisted with
   pickle. Zero external services, no API key required. Good enough for a
   few hundred documents and is what this repo ships with out of the box.

2. "chroma": Chroma vector store + OpenAI embeddings, used automatically if
   OPENAI_API_KEY is set and `chromadb`/`langchain-chroma` are installed.
   Swap to this once your knowledge base grows past ~1000 chunks or you need
   semantic (not just keyword-ish) matching.

Do NOT put fast-changing facts (weather, live prices) in here - that's what
the existing weather/place-search/currency tools are for.
"""

import glob
import os
import pickle
from dataclasses import dataclass
from pathlib import Path


@dataclass
class RetrievedChunk:
    text: str
    source: str
    score: float


class TravelKnowledgeBase:
    def __init__(self, data_dir: str = "rag/data", index_path: str = "rag/index/tfidf_index.pkl"):
        self.data_dir = data_dir
        self.index_path = index_path
        self._backend = None
        self._vectorizer = None
        self._matrix = None
        self._chunks: list[RetrievedChunk] = []
        self._use_chroma = bool(os.environ.get("OPENAI_API_KEY")) and self._chroma_available()

    @staticmethod
    def _chroma_available() -> bool:
        try:
            import langchain_chroma  # noqa: F401
            return True
        except ImportError:
            return False

    # ------------------------------------------------------------------ #
    # Build / load
    # ------------------------------------------------------------------ #
    def build(self, chunk_size: int = 800, chunk_overlap: int = 100):
        """Ingest every .md/.txt file under data_dir. Run this once offline
        (see rag/build_knowledge_base.py), not on every request."""
        docs = self._load_documents()
        chunks = self._split(docs, chunk_size, chunk_overlap)

        if self._use_chroma:
            self._build_chroma(chunks)
        else:
            self._build_tfidf(chunks)

    def _load_documents(self) -> list[tuple[str, str]]:
        """Returns list of (source_name, text)."""
        docs = []
        for path in sorted(glob.glob(os.path.join(self.data_dir, "**/*.*"), recursive=True)):
            if path.endswith((".md", ".txt")):
                with open(path, "r", encoding="utf-8") as f:
                    docs.append((os.path.basename(path), f.read()))
        return docs

    @staticmethod
    def _split(docs: list[tuple[str, str]], chunk_size: int, overlap: int) -> list[RetrievedChunk]:
        chunks = []
        for source, text in docs:
            text = text.strip()
            start = 0
            while start < len(text):
                end = min(start + chunk_size, len(text))
                chunk_text = text[start:end]
                chunks.append(RetrievedChunk(text=chunk_text, source=source, score=0.0))
                if end == len(text):
                    break
                start = end - overlap
        return chunks

    def _build_tfidf(self, chunks: list[RetrievedChunk]):
        from sklearn.feature_extraction.text import TfidfVectorizer

        self._backend = "tfidf"
        self._chunks = chunks
        self._vectorizer = TfidfVectorizer(stop_words="english")
        self._matrix = self._vectorizer.fit_transform([c.text for c in chunks])

        Path(self.index_path).parent.mkdir(parents=True, exist_ok=True)
        with open(self.index_path, "wb") as f:
            pickle.dump({"vectorizer": self._vectorizer, "matrix": self._matrix, "chunks": self._chunks}, f)

    def _build_chroma(self, chunks: list[RetrievedChunk]):
        from langchain_chroma import Chroma
        from langchain_openai import OpenAIEmbeddings
        from langchain_core.documents import Document

        self._backend = "chroma"
        embeddings = OpenAIEmbeddings()
        documents = [Document(page_content=c.text, metadata={"source": c.source}) for c in chunks]
        self._chroma_store = Chroma.from_documents(
            documents, embeddings, persist_directory="rag/index/chroma_db"
        )

    def load(self):
        """Load a previously built index. Falls back to build() if none exists."""
        if self._use_chroma:
            try:
                from langchain_chroma import Chroma
                from langchain_openai import OpenAIEmbeddings

                self._backend = "chroma"
                embeddings = OpenAIEmbeddings()
                self._chroma_store = Chroma(
                    persist_directory="rag/index/chroma_db", embedding_function=embeddings
                )
                return
            except Exception:
                pass  # fall through to tfidf

        if os.path.exists(self.index_path):
            with open(self.index_path, "rb") as f:
                data = pickle.load(f)
            self._backend = "tfidf"
            self._vectorizer = data["vectorizer"]
            self._matrix = data["matrix"]
            self._chunks = data["chunks"]
        else:
            self.build()

    # ------------------------------------------------------------------ #
    # Query
    # ------------------------------------------------------------------ #
    def query(self, question: str, k: int = 4) -> list[RetrievedChunk]:
        if self._backend is None:
            self.load()

        if self._backend == "chroma":
            docs = self._chroma_store.similarity_search_with_relevance_scores(question, k=k)
            return [RetrievedChunk(text=d.page_content, source=d.metadata.get("source", "?"), score=s) for d, s in docs]

        from sklearn.metrics.pairwise import cosine_similarity

        if self._matrix is None or self._matrix.shape[0] == 0:
            return []
        query_vec = self._vectorizer.transform([question])
        sims = cosine_similarity(query_vec, self._matrix)[0]
        top_idx = sims.argsort()[::-1][:k]
        results = []
        for i in top_idx:
            if sims[i] <= 0:
                continue
            c = self._chunks[i]
            results.append(RetrievedChunk(text=c.text, source=c.source, score=float(sims[i])))
        return results
