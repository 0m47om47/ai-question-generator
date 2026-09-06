"""
Slot-Based Question Generation Pipeline

Every run generates 17 NEW questions per game (10 city + 7 category).
Existing questions are NEVER skipped or replaced — they are only used
for duplicate checking.

Features:
- Rejected topics tracked and passed to LLM so it generates different facts
- Exponential backoff for 503/429 transient errors
- Detailed per-question logging with scores

Usage:
    python scripts/generate_questions.py
    python scripts/generate_questions.py --games quiz --dry-run
"""

import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from dotenv import load_dotenv
from pymongo import MongoClient
from sentence_transformers import SentenceTransformer
import numpy as np

from config import (
    GAMES, GAME_COLLECTION_MAP, REQUIRED_SLOTS,
    EMBEDDING_MODEL, GEMINI_MODEL, MAX_DUPLICATE_RETRIES,
    EMBEDDINGS_COLLECTION,
)
from services.ai_generator import generate_batch
from services.tavily_search import research_slots
from services.validator import validate_question, validate_slot, to_game_format
from services.duplicate_checker import DuplicateChecker
from services.question_saver import get_next_question_id, save_questions_batch

load_dotenv()

MONGODB_URI = os.getenv("MONGODB_URI")
MONGODB_DATABASE = os.getenv("MONGODB_DATABASE", "quizappdb")
SIMILARITY_THRESHOLD = float(os.getenv("SIMILARITY_THRESHOLD", "0.90"))

# Exponential backoff delays (seconds) for 503/429 errors
BACKOFF_DELAYS = [5, 15, 30]


# ── Helpers ─────────────────────────────────────────────────────────

def _extract_question_text(items) -> str:
    return items[0] if items else ""


def _extract_option_texts(items) -> list[str]:
    result = []
    for o in items[1:]:
        if isinstance(o, dict):
            result.append(o.get("text", ""))
        else:
            result.append(str(o))
    return result


def _collect_existing_question_texts(db, collection_name: str) -> list[str]:
    """Collect all question texts from a game collection (for LLM dedup hint)."""
    texts = []
    for doc in db[collection_name].find():
        for q in doc.get("questions", []):
            items = q.get("items", [])
            if items:
                texts.append(items[0])
    return texts


def _cosine_sim(a, b) -> float:
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))


def _build_embedding_text_for_check(game, slot_type, slot_value, question, options):
    city = slot_value if slot_type == "city" else ""
    category = slot_value if slot_type == "category" else ""
    lines = [
        f"Game: {game}",
        f"Type: {slot_type}",
        f"City: {city}" if city else "City:",
        f"Category: {category}" if category else "Category:",
        f"Question: {question}",
        f"Options: {', '.join(options)}",
    ]
    return "\n".join(lines)


# ── Main pipeline ──────────────────────────────────────────────────

def run_pipeline(
    games: list[str] | None = None,
    dry_run: bool = False,
):
    """
    Generate 17 NEW questions per game on every run.
    Can be called by a future scheduler.
    """
    if not MONGODB_URI:
        print("ERROR: MONGODB_URI is not set in .env")
        sys.exit(1)

    target_games = games or GAMES

    print("=" * 65)
    print("  Question Generation Pipeline")
    print("=" * 65)
    print(f"  Games       : {', '.join(target_games)}")
    print(f"  New per game: {len(REQUIRED_SLOTS)}")
    print(f"  Dry run     : {dry_run}")
    print("=" * 65)

    # ── Connect ──
    print("\nConnecting to MongoDB …")
    client = MongoClient(MONGODB_URI)
    db = client[MONGODB_DATABASE]
    client.admin.command("ping")
    print("✓ Connected\n")

    # ── Load embedding model ──
    print("Loading embedding model …")
    emb_model = SentenceTransformer(EMBEDDING_MODEL)
    print("✓ Model loaded\n")

    # ── Init duplicate checker ──
    checker = DuplicateChecker(db, emb_model, threshold=SIMILARITY_THRESHOLD)

    # ── Per-game totals ──
    grand_totals = {}

    for game in target_games:
        col_name = GAME_COLLECTION_MAP[game]
        print("=" * 65)
        print(f"  {game.upper()}")
        print("=" * 65)

        # ── Collect existing question texts (for LLM dedup hint only) ──
        existing_qtexts = _collect_existing_question_texts(db, col_name)
        print(f"\n  Existing questions in DB: {len(existing_qtexts)}")

        # ── Always target all 17 slots for generation ──
        pending_slots = list(REQUIRED_SLOTS)
        print(f"  Generating: {len(pending_slots)} new questions")

        next_id = get_next_question_id(db, col_name)
        accepted_this_run = 0
        rejected_dup_count = 0
        api_failure_count = 0
        accepted_embeddings = []       # (embedding_vector, question_text)
        accepted_game_docs = []        # collected in memory, saved as ONE doc at end
        rejected_topics: dict[tuple[str, str], list[str]] = {}  # slot → rejected texts
        used_search_topics: dict[tuple[str, str], list[str]] = {}  # track Tavily topic diversity
        request_num = 0
        consecutive_503 = 0

        for attempt in range(1, MAX_DUPLICATE_RETRIES + 1):
            if not pending_slots:
                break

            request_num += 1
            print(f"\n  Request #{request_num}")
            print(f"  Pending: {len(pending_slots)} slots")

            # ── Step 1: Tavily Research ──
            print(f"\n  ── Tavily Research ──")
            research_context = research_slots(
                slots=pending_slots,
                rejected_topics=rejected_topics,
                used_search_topics=used_search_topics,
            )
            researched_count = len(research_context)
            print(f"  Researched: {researched_count}/{len(pending_slots)} slots")

            # ── Step 2: Gemini Generation ──
            print(f"\n  ── Gemini Generation ──")
            batch = generate_batch(
                game=game,
                slots=pending_slots,
                existing_questions=existing_qtexts,
                rejected_topics=rejected_topics,
                research_context=research_context,
                model_name=GEMINI_MODEL,
            )

            # ── Handle 503 / transient errors with backoff ──
            if batch == "503":
                backoff_idx = min(consecutive_503, len(BACKOFF_DELAYS) - 1)
                delay = BACKOFF_DELAYS[backoff_idx]
                consecutive_503 += 1
                api_failure_count += 1
                if consecutive_503 > len(BACKOFF_DELAYS):
                    print(f"  ✗ API unavailable after {len(BACKOFF_DELAYS)} retries. Stopping {game}.")
                    break
                print(f"  ⏳ Retrying in {delay}s (backoff {consecutive_503}/{len(BACKOFF_DELAYS)}) …")
                time.sleep(delay)
                continue

            if batch is None:
                api_failure_count += 1
                print(f"  ✗ LLM request failed, retrying in 3s …")
                time.sleep(3)
                continue

            # Reset backoff counter on success
            consecutive_503 = 0

            print(f"  Returned : {len(batch)} items")

            # ── Validate & dedup each item ──
            accepted_slots = []
            rejected_slots = []

            for item in batch:
                slot_type = item.get("slot_type", "").strip().lower()
                slot_value = item.get("slot_value", "").strip()
                slot_key = (slot_type, slot_value)
                label = f"{slot_type}: {slot_value}"
                q_text_raw = item.get("question", "")[:60]

                # Is this slot still pending?
                if slot_key not in pending_slots:
                    continue

                # Already accepted this slot in this batch?
                if slot_key in accepted_slots:
                    continue

                # Validate slot assignment
                ok, err = validate_slot(item, slot_type, slot_value)
                if not ok:
                    print(f"    ✗ {label} — slot mismatch: {err}")
                    rejected_slots.append(slot_key)
                    continue

                # Validate game-specific schema
                ok, err = validate_question(game, item)
                if not ok:
                    print(f"    ✗ {label} — validation: {err}")
                    rejected_slots.append(slot_key)
                    continue

                # Build game doc
                game_doc = to_game_format(game, item, next_id)
                q_text = _extract_question_text(game_doc.get("items", []))
                options = _extract_option_texts(game_doc.get("items", []))

                # Check DB duplicate via vector search
                dup_result = checker.check(
                    question=q_text,
                    game=game,
                    city=slot_value if slot_type == "city" else None,
                    category=slot_value if slot_type == "category" else None,
                )
                if dup_result["is_duplicate"]:
                    score = dup_result["best_score"]
                    rejected_dup_count += 1
                    rejected_topics.setdefault(slot_key, []).append(q_text)
                    matched_q = dup_result["matches"][0]["question"] if dup_result["matches"] else "?"
                    print(f"    ❌ Duplicate")
                    print(f"       Generated: {q_text}")
                    print(f"       Existing : {matched_q}")
                    print(f"       Score    : {score:.3f}")
                    rejected_slots.append(slot_key)
                    continue

                # Check within-batch duplicate
                emb_text = _build_embedding_text_for_check(
                    game, slot_type, slot_value, q_text, options
                )
                new_emb = emb_model.encode(emb_text)
                is_batch_dup = False
                for prev_emb, prev_q in accepted_embeddings:
                    sim = _cosine_sim(new_emb, prev_emb)
                    if sim >= SIMILARITY_THRESHOLD:
                        rejected_dup_count += 1
                        rejected_topics.setdefault(slot_key, []).append(q_text)
                        print(f"    ❌ Duplicate (within batch)")
                        print(f"       Generated: {q_text}")
                        print(f"       Existing : {prev_q}")
                        print(f"       Score    : {sim:.3f}")
                        is_batch_dup = True
                        break
                if is_batch_dup:
                    rejected_slots.append(slot_key)
                    continue

                # ── Accepted! ──
                accepted_game_docs.append(game_doc)
                accepted_embeddings.append((new_emb, q_text))
                existing_qtexts.append(q_text)
                accepted_slots.append(slot_key)
                next_id += 1
                accepted_this_run += 1

                if dry_run:
                    print(f"    ✓ {label} — accepted (dry-run, id={game_doc['id']}): {q_text[:55]}")
                else:
                    print(f"    ✓ {label} — accepted (id={game_doc['id']}): {q_text[:55]}")

            # ── Update pending = remove accepted ──
            pending_slots = [s for s in pending_slots if s not in accepted_slots]

            print(f"\n  Accepted : {len(accepted_slots)}")
            print(f"  Rejected : {len(rejected_slots)}")
            if pending_slots:
                print(f"  Still pending: {len(pending_slots)}")
                for s in pending_slots:
                    rej_count = len(rejected_topics.get(s, []))
                    suffix = f" ({rej_count} rejected topics)" if rej_count else ""
                    print(f"    - {s[0]}: {s[1]}{suffix}")
            else:
                print(f"  ✓ All 17 new questions generated!")

            if pending_slots:
                time.sleep(1)

        # ── Save all accepted questions as ONE document ──
        if accepted_game_docs and not dry_run:
            print(f"\n  Saving {len(accepted_game_docs)} accepted questions as ONE document …")
            save_result = save_questions_batch(db, emb_model, game, accepted_game_docs)
            doc_id = save_result["game_doc_id"]
            print(f"  ✓ Saved! Document ID: {doc_id}")
            print(f"    totalQuestions: {save_result['count']}")
            print(f"    Embeddings created: {len(save_result['embedding_doc_ids'])}")
        elif accepted_game_docs and dry_run:
            print(f"\n  (dry-run) Would save {len(accepted_game_docs)} questions as ONE document")
        else:
            print(f"\n  No questions accepted for {game}.")

        # ── Game summary ──
        grand_totals[game] = {
            "target": len(REQUIRED_SLOTS),
            "saved": accepted_this_run,
            "rejected_duplicates": rejected_dup_count,
            "api_failures": api_failure_count,
            "remaining": len(pending_slots),
        }

        print(f"\n  ── {game.upper()} Summary ──")
        print(f"  Generated target   : {len(REQUIRED_SLOTS)}")
        print(f"  Accepted           : {accepted_this_run}")
        print(f"  Rejected duplicates: {rejected_dup_count}")
        print(f"  API failures       : {api_failure_count}")
        print(f"  Total new added    : {accepted_this_run}")

        if pending_slots:
            print(f"  ⚠ {len(pending_slots)} slots unfilled after {MAX_DUPLICATE_RETRIES} attempts")

    # ── Grand summary ──
    print("\n" + "=" * 65)
    print("  FINAL SUMMARY")
    print("=" * 65)
    total_target = 0
    total_saved = 0
    total_rej = 0
    total_api = 0
    for g, t in grand_totals.items():
        status = "✓" if t["saved"] == t["target"] else "⚠"
        print(f"  {status} {g:20s} {t['saved']}/{t['target']} new  |  dup={t['rejected_duplicates']}  api_err={t['api_failures']}")
        total_target += t["target"]
        total_saved += t["saved"]
        total_rej += t["rejected_duplicates"]
        total_api += t["api_failures"]

    print(f"\n  Generated target    : {total_target}")
    print(f"  Accepted            : {total_saved}")
    print(f"  Rejected duplicates : {total_rej}")
    print(f"  API failures        : {total_api}")
    print(f"  Total new added     : {total_saved}")
    print("=" * 65)

    client.close()
    return grand_totals


# ── CLI ─────────────────────────────────────────────────────────────

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Generate new quiz questions")
    parser.add_argument(
        "--games", nargs="+", choices=GAMES, default=None,
        help="Subset of games (default: all 4)",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Generate & validate but don't save to DB",
    )
    args = parser.parse_args()
    run_pipeline(games=args.games, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
