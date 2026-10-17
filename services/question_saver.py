"""
Question Saver — persists validated questions to the correct game
collection AND saves embeddings to question-embeddings.

Batch mode: all accepted questions go into ONE game-collection document.
Each question still gets its own embedding record for vector search.
"""

from datetime import datetime, timedelta, timezone

from sentence_transformers import SentenceTransformer

from config import GAME_COLLECTION_MAP, EMBEDDINGS_COLLECTION


def _build_embedding_text(game: str, game_doc: dict) -> str:
    """
    Same contextual format as Phase 1 embed_existing_questions.py.
    Does NOT include the answer (for semantic duplicate detection).
    """
    q_type = game_doc.get("type", "")
    city = game_doc.get("city", "")
    category = game_doc.get("category", "")
    items = game_doc.get("items", [])

    question_text = items[0] if items else ""

    # Extract option texts (handle This-or-That {text:...} objects)
    raw_options = items[1:]
    options = []
    for o in raw_options:
        if isinstance(o, dict):
            options.append(o.get("text", ""))
        else:
            options.append(str(o))

    lines = [
        f"Game: {game}",
        f"Type: {q_type}",
        f"City: {city}" if city else "City:",
        f"Category: {category}" if category else "Category:",
        f"Question: {question_text}",
        f"Options: {', '.join(options)}",
    ]
    return "\n".join(lines)


def get_next_question_id(db, collection_name: str) -> int:
    """Find the highest question id across all docs and return max+1."""
    pipeline = [
        {"$unwind": "$questions"},
        {"$group": {"_id": None, "maxId": {"$max": "$questions.id"}}},
    ]
    result = list(db[collection_name].aggregate(pipeline))
    return (result[0]["maxId"] + 1) if result else 1


def save_questions_batch(
    db,
    model: SentenceTransformer,
    game: str,
    game_docs: list[dict],
) -> dict:
    """
    Save ALL accepted questions from one generation run as:
      1. ONE document in the game collection (questions array)
      2. One embedding record per question in question-embeddings

    Args:
        db:         pymongo database
        model:      SentenceTransformer model for embeddings
        game:       game name (quiz, this-that, etc.)
        game_docs:  list of question dicts in game-collection format

    Returns:
        {"game_doc_id": ..., "embedding_doc_ids": [...], "count": int}
    """
    if not game_docs:
        return {"game_doc_id": None, "embedding_doc_ids": [], "count": 0}

    collection_name = GAME_COLLECTION_MAP[game]
    now = (datetime.now(timezone.utc) + timedelta(days=1)).isoformat()

    # ── 1. Insert ONE document with all questions ──
    game_insert = {
        "createdAt": now,
        "source": "ai-generated",
        "totalQuestions": len(game_docs),
        "questions": game_docs,
    }
    game_result = db[collection_name].insert_one(game_insert)
    game_doc_id = game_result.inserted_id

    # ── 2. Insert individual embedding records ──
    embedding_doc_ids = []
    for game_doc in game_docs:
        items = game_doc.get("items", [])
        question_text = items[0] if items else ""

        raw_options = items[1:]
        options = []
        for o in raw_options:
            if isinstance(o, dict):
                options.append(o.get("text", ""))
            else:
                options.append(str(o))

        emb_text = _build_embedding_text(game, game_doc)
        embedding = model.encode(emb_text).tolist()

        emb_doc = {
            "game": game,
            "type": game_doc.get("type", ""),
            "city": game_doc.get("city") or None,
            "category": game_doc.get("category") or None,
            "question": question_text,
            "options": options,
            "answer": game_doc.get("answer", ""),
            "embedding": embedding,
            "source": "ai-generated",
            "sourceCollection": collection_name,
            "sourceDocumentId": game_doc_id,
            "sourceQuestionId": game_doc.get("id"),
            "createdAt": now,
        }
        emb_result = db[EMBEDDINGS_COLLECTION].insert_one(emb_doc)
        embedding_doc_ids.append(emb_result.inserted_id)

    return {
        "game_doc_id": game_doc_id,
        "embedding_doc_ids": embedding_doc_ids,
        "count": len(game_docs),
    }
