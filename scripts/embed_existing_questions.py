"""
Phase 1 — Embed all existing quiz questions into a new MongoDB collection.

Reads questions from the four game collections, generates 384-dim embeddings
using the local sentence-transformers model (all-MiniLM-L6-v2), and stores
them in the `question-embeddings` collection.

Safe to run multiple times — duplicates are detected via
(sourceCollection, sourceDocumentId, sourceQuestionId).

Usage:
    python scripts/embed_existing_questions.py
"""

import os
import sys
import time
from datetime import datetime, timezone

from dotenv import load_dotenv
from pymongo import MongoClient
from sentence_transformers import SentenceTransformer

# ── Load environment ────────────────────────────────────────────────
load_dotenv()

MONGODB_URI = os.getenv("MONGODB_URI")
MONGODB_DATABASE = os.getenv("MONGODB_DATABASE", "quizappdb")

if not MONGODB_URI:
    print("ERROR: MONGODB_URI is not set in .env")
    sys.exit(1)

# ── Collection → game-name mapping ─────────────────────────────────
COLLECTION_MAP = {
    "quiz-game":            "quiz",
    "this-that-game":       "this-that",
    "guessthenumber-game":  "guessthenumber",
    "wrongansweronly-game":  "wrongansweronly",
}

EMBEDDINGS_COLLECTION = "question-embeddings"


# ── Helpers ─────────────────────────────────────────────────────────

def extract_option_text(option):
    """Return plain text from an option (handles This-or-That objects)."""
    if isinstance(option, dict):
        return option.get("text", "")
    return str(option)


def build_embedding_text(game: str, q: dict) -> str:
    """
    Build a contextual text for embedding.

    Includes game, type, city/category, question, and options —
    but NOT the answer (for semantic duplicate detection later).
    """
    q_type = q.get("type", "")
    city = q.get("city", "")
    category = q.get("category", "")
    items = q.get("items", [])

    question_text = items[0] if items else ""
    options = [extract_option_text(o) for o in items[1:]]

    lines = [
        f"Game: {game}",
        f"Type: {q_type}",
        f"City: {city}" if city else "City:",
        f"Category: {category}" if category else "Category:",
        f"Question: {question_text}",
        f"Options: {', '.join(options)}",
    ]
    return "\n".join(lines)


def extract_question_fields(game: str, q: dict) -> dict:
    """
    Pull the normalised fields we store alongside the embedding.
    """
    items = q.get("items", [])
    question_text = items[0] if items else ""
    options = [extract_option_text(o) for o in items[1:]]

    return {
        "game":     game,
        "type":     q.get("type", ""),
        "city":     q.get("city") or None,
        "category": q.get("category") or None,
        "question": question_text,
        "options":  options,
        "answer":   q.get("answer", ""),
    }


# ── Main migration ─────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("Phase 1 — Embed Existing Questions")
    print("=" * 60)

    # --- Connect to MongoDB ---
    print(f"\nConnecting to MongoDB … ({MONGODB_DATABASE})")
    client = MongoClient(MONGODB_URI)
    db = client[MONGODB_DATABASE]

    # Quick connectivity check
    try:
        client.admin.command("ping")
        print("✓ MongoDB connection successful\n")
    except Exception as e:
        print(f"✗ MongoDB connection failed: {e}")
        sys.exit(1)

    embeddings_col = db[EMBEDDINGS_COLLECTION]

    # --- Load embedding model (local, free) ---
    print("Loading embedding model (all-MiniLM-L6-v2) …")
    model = SentenceTransformer("all-MiniLM-L6-v2")
    print("✓ Model loaded\n")

    # --- Build set of already-embedded source keys for dedup ---
    print("Checking for already-embedded questions …")
    existing_keys = set()
    for doc in embeddings_col.find(
        {"source": "existing"},
        {"sourceCollection": 1, "sourceDocumentId": 1, "sourceQuestionId": 1},
    ):
        key = (
            doc.get("sourceCollection"),
            str(doc.get("sourceDocumentId")),
            doc.get("sourceQuestionId"),
        )
        existing_keys.add(key)
    print(f"  Found {len(existing_keys)} already-embedded questions\n")

    # --- Iterate over game collections ---
    stats = {
        "total_found": 0,
        "already_embedded": 0,
        "newly_embedded": 0,
        "failed": 0,
    }

    for collection_name, game_name in COLLECTION_MAP.items():
        col = db[collection_name]
        doc_count = col.count_documents({})
        print(f"─── {collection_name} ({doc_count} documents) ───")

        for doc in col.find():
            doc_id = doc["_id"]
            questions = doc.get("questions", [])

            for q in questions:
                q_id = q.get("id")
                stats["total_found"] += 1

                # ── Duplicate check ──
                source_key = (collection_name, str(doc_id), q_id)
                if source_key in existing_keys:
                    stats["already_embedded"] += 1
                    continue

                try:
                    # ── Build fields & embedding text ──
                    fields = extract_question_fields(game_name, q)
                    emb_text = build_embedding_text(game_name, q)
                    embedding = model.encode(emb_text).tolist()

                    # ── Assemble embedding document ──
                    emb_doc = {
                        **fields,
                        "embedding": embedding,
                        "source": "existing",
                        "sourceCollection": collection_name,
                        "sourceDocumentId": doc_id,
                        "sourceQuestionId": q_id,
                        "createdAt": datetime.now(timezone.utc).isoformat(),
                    }

                    embeddings_col.insert_one(emb_doc)
                    stats["newly_embedded"] += 1

                    # Progress indicator
                    items = q.get("items", [])
                    q_text = (items[0][:60] + "…") if items and len(items[0]) > 60 else (items[0] if items else "?")
                    print(f"  ✓ [{game_name}] id={q_id}  {q_text}")

                except Exception as e:
                    stats["failed"] += 1
                    print(f"  ✗ [{game_name}] id={q_id}  ERROR: {e}")

        print()

    # --- Summary ---
    print("=" * 60)
    print("Migration complete!")
    print("=" * 60)
    print(f"  Total questions found : {stats['total_found']}")
    print(f"  Already embedded      : {stats['already_embedded']}")
    print(f"  Newly embedded        : {stats['newly_embedded']}")
    print(f"  Failed                : {stats['failed']}")
    print(f"\n  Collection: {MONGODB_DATABASE}.{EMBEDDINGS_COLLECTION}")
    print("=" * 60)

    client.close()


if __name__ == "__main__":
    main()
