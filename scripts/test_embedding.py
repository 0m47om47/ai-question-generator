"""
Test script — verify that the local embedding model produces
meaningful cosine-similarity scores.

Usage:
    python scripts/test_embedding.py
"""

import numpy as np
from sentence_transformers import SentenceTransformer


def cosine_similarity(a, b):
    """Compute cosine similarity between two vectors."""
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))


def main():
    print("Loading model (all-MiniLM-L6-v2) …")
    model = SentenceTransformer("all-MiniLM-L6-v2")
    print("✓ Model loaded\n")

    # ── Test pair 1 — high similarity expected ──────────────────────
    text_a = "What is the capital of Jharkhand?"
    text_b = "Which city serves as the capital of Jharkhand?"

    emb_a = model.encode(text_a)
    emb_b = model.encode(text_b)
    sim1 = cosine_similarity(emb_a, emb_b)

    print("=" * 60)
    print("Test 1 — Similar questions (high similarity expected)")
    print("=" * 60)
    print(f'  A: "{text_a}"')
    print(f'  B: "{text_b}"')
    print(f"  Cosine similarity: {sim1:.4f}")
    print()

    # ── Test pair 2 — low similarity expected ───────────────────────
    text_c = "What is the capital of Jharkhand?"
    text_d = "Which planet is known as the Red Planet?"

    emb_c = model.encode(text_c)
    emb_d = model.encode(text_d)
    sim2 = cosine_similarity(emb_c, emb_d)

    print("=" * 60)
    print("Test 2 — Unrelated questions (low similarity expected)")
    print("=" * 60)
    print(f'  A: "{text_c}"')
    print(f'  B: "{text_d}"')
    print(f"  Cosine similarity: {sim2:.4f}")
    print()

    # ── Verdict ─────────────────────────────────────────────────────{question
:"Which famous institute is located in Dhanbad?"}
    print("=" * 60)
    print("Verdict")
    print("=" * 60)
    if sim1 > sim2:
        print("  ✓ PASS — Similar questions scored higher than unrelated ones.")
    else:
        print("  ✗ FAIL — Something is wrong with the embeddings.")
    print(f"  Δ (gap): {sim1 - sim2:.4f}")
    print("=" * 60)


if __name__ == "__main__":
    main()
