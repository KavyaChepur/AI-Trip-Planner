"""
Run this once (or whenever rag/data/*.md changes) to (re)build the RAG index:

    python -m rag.build_knowledge_base

This is intentionally NOT run on every request - embedding/indexing is a
batch/offline step, not something you want in the hot path of a user query.
"""

from utils.knowledge_base import TravelKnowledgeBase


def main():
    kb = TravelKnowledgeBase()
    kb.build()
    print(f"Knowledge base built using backend: {kb._backend}")
    print(f"Indexed {len(kb._chunks) if kb._backend == 'tfidf' else 'N/A'} chunks.")

    # quick smoke test
    results = kb.query("best time to visit Goa", k=2)
    print("\nSample query -> 'best time to visit Goa':")
    for r in results:
        print(f"  [{r.source} | score={r.score:.3f}] {r.text[:120]}...")


if __name__ == "__main__":
    main()
