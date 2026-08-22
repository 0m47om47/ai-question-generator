"""
Phase 2 — Create MongoDB Atlas Vector Search index on question-embeddings.

Creates a vector search index named `question_vector_index` with:
  - Vector field: embedding (384 dimensions, cosine similarity)
  - Filter fields: game, type, city, category

Usage:
    python scripts/create_vector_index.py
"""

import os
import sys
import time

from dotenv import load_dotenv
from pymongo import MongoClient
from pymongo.operations import SearchIndexModel

# ── Load environment ────────────────────────────────────────────────
load_dotenv()

MONGODB_URI = os.getenv("MONGODB_URI")
MONGODB_DATABASE = os.getenv("MONGODB_DATABASE", "quizappdb")

if not MONGODB_URI:
    print("ERROR: MONGODB_URI is not set in .env")
    sys.exit(1)

EMBEDDINGS_COLLECTION = "question-embeddings"
INDEX_NAME = "question_vector_index"


def main():
    print("=" * 60)
    print("Phase 2 — Create Atlas Vector Search Index")
    print("=" * 60)

    
    print(f"\nConnecting to MongoDB … ({MONGODB_DATABASE})")
    client = MongoClient(MONGODB_URI)
    db = client[MONGODB_DATABASE]

    try:
        client.admin.command("ping")
        print("MongoDB connection successful\n")
    except Exception as e:
        print(f"✗ MongoDB connection failed: {e}")
        sys.exit(1)

    collection = db[EMBEDDINGS_COLLECTION]

    # --- Check if index already exists ---
    print(f"Checking for existing index '{INDEX_NAME}' …")
    existing_indexes = list(collection.list_search_indexes())
    for idx in existing_indexes:
        if idx.get("name") == INDEX_NAME:
            print(f"✓ Index '{INDEX_NAME}' already exists. No action needed.")
            print(f"  Status: {idx.get('status', 'unknown')}")
            client.close()
            return

    # --- Create Vector Search Index ---
    print(f"Creating vector search index '{INDEX_NAME}' …\n")

    search_index_model = SearchIndexModel(
        definition={
            "fields": [
                {
                    "type": "vector",
                    "path": "embedding",
                    "numDimensions": 384,
                    "similarity": "cosine",
                },
                {
                    "type": "filter",
                    "path": "game",
                },
                {
                    "type": "filter",
                    "path": "type",
                },
                {
                    "type": "filter",
                    "path": "city",
                },
                {
                    "type": "filter",
                    "path": "category",
                },
            ]
        },
        name=INDEX_NAME,
        type="vectorSearch",
    )

    result = collection.create_search_index(model=search_index_model)
    print(f"✓ Index creation initiated: {result}\n")

    # --- Wait for index to become ready ---
    print("Waiting for index to become ready …")
    max_wait = 120  # seconds
    start = time.time()
    while time.time() - start < max_wait:
        indexes = list(collection.list_search_indexes())
        for idx in indexes:
            if idx.get("name") == INDEX_NAME:
                status = idx.get("status", "unknown")
                print(f"  Status: {status}")
                if status == "READY":
                    print(f"\n✓ Index '{INDEX_NAME}' is READY!")
                    print("=" * 60)
                    _print_index_summary(idx)
                    print("=" * 60)
                    client.close()
                    return
        time.sleep(5)

    print(f"\n⚠ Index not ready after {max_wait}s. It may still be building.")
    print("  Check Atlas UI or run this script again to verify status.")
    print("=" * 60)
    client.close()


def _print_index_summary(idx: dict):
    """Print a human-readable summary of the index."""
    print(f"\n  Index name      : {idx.get('name')}")
    print(f"  Status          : {idx.get('status')}")
    definition = idx.get("latestDefinition", idx.get("definition", {}))
    fields = definition.get("fields", [])
    for f in fields:
        ftype = f.get("type", "")
        path = f.get("path", "")
        if ftype == "vector":
            dims = f.get("numDimensions", "?")
            sim = f.get("similarity", "?")
            print(f"  Vector field    : {path} ({dims}D, {sim})")
        elif ftype == "filter":
            print(f"  Filter field    : {path}")


if __name__ == "__main__":
    main()
