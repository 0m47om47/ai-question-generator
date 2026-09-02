"""
Question Saver — persists validated questions to the correct game
collection AND saves the embedding to question-embeddings.
"""

from datetime import datetime, timezone

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


def save_question(
    db,
    model: SentenceTransformer,
    game: str,
    game_doc: dict,
) -> dict:
    """
    Save a validated question to:
      1. The correct game collection (new document with questions array)
      2. The question-embeddings collection (with embedding vector)

    Returns {"game_doc_id": ..., "embedding_doc_id": ...}
    """
    collection_name = GAME_COLLECTION_MAP[game]
    now = datetime.now(timezone.utc).isoformat()

    # ── 1. Save to game collection ──
    game_insert = {
        "createdAt": now,
        "source": "ai-generated",
        "totalQuestions": 1,
        "questions": [game_doc],
    }
    game_result = db[collection_name].insert_one(game_insert)

    # ── 2. Build & save embedding ──
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
        "sourceDocumentId": game_result.inserted_id,
        "sourceQuestionId": game_doc.get("id"),
        "createdAt": now,
    }
    emb_result = db[EMBEDDINGS_COLLECTION].insert_one(emb_doc)

    return {
        "game_doc_id": game_result.inserted_id,
        "embedding_doc_id": emb_result.inserted_id,
    }
