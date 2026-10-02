"""
Duplicate Checker Service — Phase 2

Uses MongoDB Atlas Vector Search to find semantically similar questions
in the `question-embeddings` collection.

Usage (as a module):
    from services.duplicate_checker import DuplicateChecker

    checker = DuplicateChecker(db, model)
    results = checker.check("What is the capital of Jharkhand?", game="quiz")
"""

import os
from typing import Optional

import numpy as np
from sentence_transformers import SentenceTransformer


# Atlas vector search retrieval threshold (over broad metadata vector)
DEFAULT_SIMILARITY_THRESHOLD = 0.90

# Pure question text thresholds (comparing just the question sentences, without metadata):
# - >= 0.93: Definite duplicate (nearly identical wording/fact)
# - >= 0.78: Potential duplicate requiring Gemini 2-stage verification
# - < 0.78 : Completely different question (even if same city/game)
PURE_DUP_THRESHOLD = 0.93
PURE_VERIFY_THRESHOLD = 0.78

EMBEDDINGS_COLLECTION = "question-embeddings"
INDEX_NAME = "question_vector_index"


def _build_query_text(
    question: str,
    game: Optional[str] = None,
    category: Optional[str] = None,
    city: Optional[str] = None,
) -> str:
    """
    Build contextual text for the query embedding.

    Uses the same format as Phase 1 so that embeddings are comparable.
    """
    lines = [
        f"Game: {game or ''}",
        f"Type: {'city' if city else ('category' if category else '')}",
        f"City: {city or ''}",
        f"Category: {category or ''}",
        f"Question: {question}",
    ]
    return "\n".join(lines)


class DuplicateChecker:
    """
    Check whether a question is a semantic duplicate of an existing one
    using MongoDB Atlas Vector Search.
    """

    def __init__(
        self,
        db,
        model: SentenceTransformer,
        threshold: Optional[float] = None,
    ):
        self.collection = db[EMBEDDINGS_COLLECTION]
        self.model = model
        self.threshold = threshold or float(
            os.getenv("SIMILARITY_THRESHOLD", DEFAULT_SIMILARITY_THRESHOLD)
        )

    def check(
        self,
        question: str,
        game: Optional[str] = None,
        category: Optional[str] = None,
        city: Optional[str] = None,
        top_k: int = 5,
        use_filters: bool = False,
    ) -> dict:
        """
        Search for semantically similar questions.

        Args:
            question:    The question text to check.
            game:        Optional game filter (quiz, this-that, …).
            category:    Optional category filter.
            city:        Optional city filter.
            top_k:       Number of nearest neighbours to return.
            use_filters: If True, apply game/category/city as pre-filters
                         in vector search. If False, search broadly.

        Returns:
            {
                "query": str,
                "threshold": float,
                "is_duplicate": bool,
                "best_score": float | None,
                "matches": [
                    {
                        "question": str,
                        "game": str,
                        "type": str,
                        "city": str | None,
                        "category": str | None,
                        "options": list,
                        "answer": str,
                        "score": float,
                        "is_duplicate": bool,
                    },
                    …
                ]
            }
        """

        # ── Generate query embedding ──
        query_text = _build_query_text(question, game, category, city)
        query_vector = self.model.encode(query_text).tolist()

        # ── Build optional pre-filter ──
        pre_filter = {}
        if use_filters:
            filter_conditions = []
            if game:
                filter_conditions.append({"game": {"$eq": game}})
            if category:
                filter_conditions.append({"category": {"$eq": category}})
            if city:
                filter_conditions.append({"city": {"$eq": city}})
            if filter_conditions:
                pre_filter = (
                    {"$and": filter_conditions}
                    if len(filter_conditions) > 1
                    else filter_conditions[0]
                )

        # ── Vector Search aggregation ──
        vector_search_stage = {
            "$vectorSearch": {
                "index": INDEX_NAME,
                "path": "embedding",
                "queryVector": query_vector,
                "numCandidates": max(top_k * 20, 100),
                "limit": top_k,
            }
        }
        if pre_filter:
            vector_search_stage["$vectorSearch"]["filter"] = pre_filter

        project_stage = {
            "$project": {
                "question": 1,
                "game": 1,
                "type": 1,
                "city": 1,
                "category": 1,
                "options": 1,
                "answer": 1,
                "score": {"$meta": "vectorSearchScore"},
                "_id": 0,
            }
        }

        pipeline = [vector_search_stage, project_stage]
        results = list(self.collection.aggregate(pipeline))

        # ── Classify matches & compute pure question similarity ──
        matches = []
        best_score = None
        best_pure_score = None
        is_duplicate = False
        needs_verification = False

        if results:
            existing_questions = [r.get("question", "") for r in results]
            q_vec = self.model.encode(question)
            ex_vecs = self.model.encode(existing_questions)

            q_norm = np.linalg.norm(q_vec)
            ex_norms = np.linalg.norm(ex_vecs, axis=1)
            dot_prods = np.dot(ex_vecs, q_vec)
            pure_scores = dot_prods / (q_norm * ex_norms + 1e-9)

            for i, r in enumerate(results):
                score = r.pop("score", 0.0)
                pure_score = float(pure_scores[i])

                # Definite duplicate: only if the actual question wording is >= 0.93 identical
                is_dup = pure_score >= PURE_DUP_THRESHOLD

                # Needs Gemini verification: pure question similarity is moderately high (>= 0.78)
                # and Atlas vector score also flagged it (>= threshold)
                verify = (pure_score >= PURE_VERIFY_THRESHOLD) and (score >= self.threshold) and not is_dup

                if is_dup:
                    is_duplicate = True
                if verify:
                    needs_verification = True

                if best_score is None or score > best_score:
                    best_score = score
                if best_pure_score is None or pure_score > best_pure_score:
                    best_pure_score = pure_score

                matches.append({
                    **r,
                    "score": score,
                    "pure_score": pure_score,
                    "is_duplicate": is_dup,
                    "needs_verification": verify,
                })

        return {
            "query": question,
            "threshold": self.threshold,
            "is_duplicate": is_duplicate,
            "needs_verification": needs_verification,
            "best_score": best_score,
            "best_pure_score": best_pure_score,
            "matches": matches,
        }
