"""
Validator — validates AI-generated JSON against the exact game schema
(source of truth: docs/game-formats/*.json) and converts to the
MongoDB document format used by each game collection.
"""

from typing import Tuple


# ── Public API ──────────────────────────────────────────────────────

def validate_question(game: str, data: dict) -> Tuple[bool, str]:
    """
    Validate a single AI-generated question dict for a game type.
    Returns (is_valid, error_message).
    """
    if not isinstance(data, dict):
        return False, "Response is not a dict"

    _VALIDATORS = {
        "quiz":            _validate_quiz,
        "this-that":       _validate_this_that,
        "guessthenumber":  _validate_guess_number,
        "wrongansweronly":  _validate_wrong_answer,
    }
    v = _VALIDATORS.get(game)
    if not v:
        return False, f"Unknown game type: {game}"
    return v(data)


def validate_slot(data: dict, expected_type: str, expected_value: str) -> Tuple[bool, str]:
    """
    Check that the returned question matches the requested slot.
    """
    stype = data.get("slot_type", "").strip().lower()
    svalue = data.get("slot_value", "").strip()

    if stype != expected_type:
        return False, f"Expected slot_type '{expected_type}', got '{stype}'"
    if svalue.lower() != expected_value.lower():
        return False, f"Expected slot_value '{expected_value}', got '{svalue}'"
    return True, ""


def to_game_format(game: str, data: dict, q_id: int) -> dict:
    """
    Convert validated AI output into the exact MongoDB document
    format used by the existing game collections.

    Schemas match docs/game-formats/*.json exactly.
    """
    slot_type = data.get("slot_type", "").strip().lower()
    slot_value = data.get("slot_value", "").strip()

    # Base fields
    doc = {"id": q_id, "type": slot_type}
    if slot_type == "city":
        doc["city"] = slot_value
    else:
        doc["category"] = slot_value

    if game == "quiz":
        doc["items"] = [data["question"]] + data["options"]
        doc["answer"] = data["answer"]

    elif game == "this-that":
        # This-or-That options are {text: ...} objects (no img field)
        doc["items"] = [
            data["question"],
            {"text": data["option_a"]},
            {"text": data["option_b"]},
        ]
        doc["answer"] = data["answer"]

    elif game == "guessthenumber":
        doc["items"] = [data["question"]] + data["options"]
        doc["answer"] = data["answer"]

    elif game == "wrongansweronly":
        # items: [question, correct_option, wrong_option]
        # answer = wrong_option (game mechanic)
        doc["items"] = [
            data["question"],
            data["correct_option"],
            data["wrong_option"],
        ]
        doc["answer"] = data["wrong_option"]

    else:
        raise ValueError(f"Unknown game type: {game}")

    return doc


# ── Per-game validators ────────────────────────────────────────────

def _validate_quiz(d: dict) -> Tuple[bool, str]:
    q = d.get("question")
    opts = d.get("options")
    ans = d.get("answer")
    if not q or not isinstance(q, str):
        return False, "Missing/invalid 'question'"
    if not isinstance(opts, list) or len(opts) != 4:
        return False, f"Need 4 options, got {len(opts) if isinstance(opts, list) else 'N/A'}"
    if not all(isinstance(o, str) and o.strip() for o in opts):
        return False, "All options must be non-empty strings"
    if ans not in opts:
        return False, f"Answer '{ans}' not in options"
    return True, ""


def _validate_this_that(d: dict) -> Tuple[bool, str]:
    q = d.get("question")
    a = d.get("option_a")
    b = d.get("option_b")
    ans = d.get("answer")
    if not q or not isinstance(q, str):
        return False, "Missing/invalid 'question'"
    if not a or not isinstance(a, str):
        return False, "Missing/invalid 'option_a'"
    if not b or not isinstance(b, str):
        return False, "Missing/invalid 'option_b'"
    if ans not in (a, b):
        return False, f"Answer '{ans}' must match option_a or option_b"
    return True, ""


def _validate_guess_number(d: dict) -> Tuple[bool, str]:
    q = d.get("question")
    opts = d.get("options")
    ans = d.get("answer")
    if not q or not isinstance(q, str):
        return False, "Missing/invalid 'question'"
    if not isinstance(opts, list) or len(opts) != 4:
        return False, f"Need 4 options, got {len(opts) if isinstance(opts, list) else 'N/A'}"
    if not all(isinstance(o, str) and o.strip() for o in opts):
        return False, "All options must be non-empty strings"
    if ans not in opts:
        return False, f"Answer '{ans}' not in options"
    return True, ""


def _validate_wrong_answer(d: dict) -> Tuple[bool, str]:
    q = d.get("question")
    correct = d.get("correct_option")
    wrong = d.get("wrong_option")
    ans = d.get("answer")
    if not q or not isinstance(q, str):
        return False, "Missing/invalid 'question'"
    if not correct or not isinstance(correct, str):
        return False, "Missing/invalid 'correct_option'"
    if not wrong or not isinstance(wrong, str):
        return False, "Missing/invalid 'wrong_option'"
    if ans != wrong:
        return False, f"Answer must equal wrong_option ('{ans}' != '{wrong}')"
    return True, ""
