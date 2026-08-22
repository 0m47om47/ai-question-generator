"""
Phase 2 — Test MongoDB Atlas Vector Search duplicate detection.

Runs four tests:
  1. Semantic duplicate (paraphrased question)
  2. Exact duplicate (question that exists in the collection)
  3. Clearly different question
  4. Context-aware: Dhanbad question vs unrelated city question (with filter)

Usage:
    python scripts/test_vector_search.py
"""

import os
import sys

# Add project root to path so we can import services
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from dotenv import load_dotenv
from pymongo import MongoClient
from sentence_transformers import SentenceTransformer

from services.duplicate_checker import DuplicateChecker

# ── Load environment ────────────────────────────────────────────────
load_dotenv()

MONGODB_URI = os.getenv("MONGODB_URI")
MONGODB_DATABASE = os.getenv("MONGODB_DATABASE", "quizappdb")
SIMILARITY_THRESHOLD = float(os.getenv("SIMILARITY_THRESHOLD", "0.90"))

if not MONGODB_URI:
    print("ERROR: MONGODB_URI is not set in .env")
    sys.exit(1)


def print_result(test_name: str, result: dict):
    """Pretty-print a duplicate-check result."""
    print("=" * 65)
    print(f"  {test_name}")
    print("=" * 65)
    print(f"  Query       : {result['query']}")
    print(f"  Threshold   : {result['threshold']}")
    print(f"  Best score  : {result['best_score']:.4f}" if result["best_score"] else "  Best score  : N/A")
    print(f"  Duplicate?  : {'YES ⚠' if result['is_duplicate'] else 'NO ✓'}")
    print()

    if result["matches"]:
        print(f"  Top {len(result['matches'])} matches:")
        for i, m in enumerate(result["matches"], 1):
            dup_flag = " [DUPLICATE]" if m["is_duplicate"] else ""
            print(f"    {i}. score={m['score']:.4f}{dup_flag}")
            print(f"       Q: {m['question']}")
            ctx_parts = []
            if m.get("game"):
                ctx_parts.append(f"game={m['game']}")
            if m.get("city"):
                ctx_parts.append(f"city={m['city']}")
            if m.get("category"):
                ctx_parts.append(f"category={m['category']}")
            if ctx_parts:
                print(f"       ({', '.join(ctx_parts)})")
            print()
    else:
        print("  No matches found.\n")


def main():
    print("\n" + "=" * 65)
    print("  Phase 2 — Vector Search Duplicate Detection Tests")
    print("=" * 65)
    print(f"  Threshold: {SIMILARITY_THRESHOLD}\n")

    # ── Setup ──
    print("Connecting to MongoDB …")
    client = MongoClient(MONGODB_URI)
    db = client[MONGODB_DATABASE]
    client.admin.command("ping")
    print("✓ Connected\n")

    print("Loading embedding model (all-MiniLM-L6-v2) …")
    model = SentenceTransformer("all-MiniLM-L6-v2")
    print("✓ Model loaded\n")

    checker = DuplicateChecker(db, model, threshold=SIMILARITY_THRESHOLD)

    # ──────────────────────────────────────────────────────────────
    # TEST 1 — Semantic duplicate (paraphrased)
    # ──────────────────────────────────────────────────────────────
    result1 = checker.check(
        question="Which city serves as the capital of Jharkhand?",
        game="quiz",
    )
    print_result("TEST 1 — Semantic Duplicate (paraphrased)", result1)

    # ──────────────────────────────────────────────────────────────
    # TEST 2 — Exact duplicate (pick a real question from the DB)
    # ──────────────────────────────────────────────────────────────
    # Grab an actual question from question-embeddings
    sample = db["question-embeddings"].find_one(
        {"game": "quiz", "question": {"$exists": True}},
        {"question": 1, "game": 1, "city": 1, "category": 1, "_id": 0},
    )
    if sample:
        result2 = checker.check(
            question=sample["question"],
            game=sample.get("game"),
            city=sample.get("city"),
            category=sample.get("category"),
        )
        print_result("TEST 2 — Exact Duplicate (real question from DB)", result2)
    else:
        print("⚠ Could not find a sample question for Test 2.\n")

    # ──────────────────────────────────────────────────────────────
    # TEST 3 — Clearly different question
    # ──────────────────────────────────────────────────────────────
    result3 = checker.check(
        question="What is the largest planet in our solar system?",
        game="quiz",
    )
    print_result("TEST 3 — Clearly Different Question", result3)

    # ──────────────────────────────────────────────────────────────
    # TEST 4 — Context / city filter
    # ──────────────────────────────────────────────────────────────
    # Search a Dhanbad question with city filter
    result4a = checker.check(
        question="Which famous institute is located in Dhanbad?",
        game="quiz",
        city="Dhanbad",
        use_filters=True,
    )
    print_result("TEST 4a — Dhanbad question (with city filter)", result4a)

    # Same question but filtered to a different city
    result4b = checker.check(
        question="Which famous institute is located in Dhanbad?",
        game="quiz",
        city="Dumka",
        use_filters=True,
    )
    print_result("TEST 4b — Same Dhanbad question (filtered to Dumka)", result4b)

    # ──────────────────────────────────────────────────────────────
    # Summary
    # ──────────────────────────────────────────────────────────────
    print("=" * 65)
    print("  Summary")
    print("=" * 65)
    tests = [
        ("Test 1 — Semantic duplicate", result1),
        ("Test 3 — Different question",  result3),
        ("Test 4a — Dhanbad (filtered)", result4a),
        ("Test 4b — Dumka (filtered)",   result4b),
    ]
    if sample:
        tests.insert(1, ("Test 2 — Exact duplicate", checker.check(
            question=sample["question"],
            game=sample.get("game"),
            city=sample.get("city"),
            category=sample.get("category"),
        )))

    for name, r in tests:
        score = f"{r['best_score']:.4f}" if r["best_score"] else "N/A"
        dup = "YES" if r["is_duplicate"] else "NO"
        print(f"  {name:40s}  score={score}  dup={dup}")

    print("=" * 65)
    print()

    # ── Confirm no modifications ──
    emb_count = db["question-embeddings"].count_documents({})
    print(f"  question-embeddings count: {emb_count} (should be 15208)")
    for col_name in ["quiz-game", "this-that-game", "guessthenumber-game", "wrongansweronly-game"]:
        count = db[col_name].count_documents({})
        print(f"  {col_name} count: {count} (unchanged)")
    print()

    client.close()
    print("Done.\n")


if __name__ == "__main__":
    main()
