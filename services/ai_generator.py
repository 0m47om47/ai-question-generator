"""
AI Question Generator — batch-generates questions for missing slots
using Gemini with structured JSON output.

ONE LLM call per game (ideal case). Follow-up calls only for
missing / rejected / duplicate slots, with rejected topics passed
to prevent regenerating the same fact.
"""

import json
import os
import time
from typing import Optional

from google import genai
from google.genai import types


# Exponential backoff delays for 503 / transient errors
_BACKOFF_DELAYS = [5, 15, 30]


def _get_client() -> genai.Client:
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise ValueError("GOOGLE_API_KEY is not set in .env")
    return genai.Client(
        api_key=api_key,
        http_options=types.HttpOptions(
            timeout=120_000,   # 120 seconds max per request (milliseconds)
        ),
    )


# ── Prompt helpers ──────────────────────────────────────────────────

def _slots_description(slots: list[tuple[str, str]]) -> str:
    lines = []
    for stype, svalue in slots:
        if stype == "city":
            lines.append(f'  - type: "city", city: "{svalue}"')
        else:
            lines.append(f'  - type: "category", category: "{svalue}"')
    return "\n".join(lines)


def _rejected_block(rejected_topics: dict[tuple[str, str], list[str]],
                     slots: list[tuple[str, str]]) -> str:
    """Build a prompt section listing previously rejected question texts per slot."""
    lines = []
    for slot in slots:
        rejected = rejected_topics.get(slot, [])
        if rejected:
            stype, svalue = slot
            lines.append(f"\n  {stype}: {svalue} — REJECTED questions (use a COMPLETELY DIFFERENT fact/topic):")
            for r in rejected:
                lines.append(f"    ✗ {r}")
    if lines:
        return "\nPREVIOUSLY REJECTED QUESTIONS FOR THESE SLOTS:" + "\n".join(lines) + "\n"
    return ""


# ── Quiz prompt ─────────────────────────────────────────────────────

def _build_quiz_prompt(
    slots: list[tuple[str, str]],
    existing_questions: list[str],
    rejected_topics: dict[tuple[str, str], list[str]] | None = None,
) -> str:
    existing_block = ""
    if existing_questions:
        # Show up to 80 existing questions for better dedup
        sample = existing_questions[:80]
        existing_block = (
            "\n\nEXISTING QUESTIONS (do NOT repeat, paraphrase, reword, or use the same underlying fact):\n"
            + "\n".join(f"- {q}" for q in sample)
        )

    rejected_block = _rejected_block(rejected_topics or {}, slots)

    return f"""Generate quiz questions for the following slots.

SLOTS TO FILL:
{_slots_description(slots)}

CRITICAL DIVERSITY RULES:
- Generate EXACTLY one question per slot.
- Every question must test a GENUINELY DIFFERENT fact, event, person, place, concept, or statistic.
- For CITY slots: use a lesser-known or surprising fact about that city. Do NOT ask about the most famous landmark, river, or industry that is likely already covered. Explore local culture, festivals, cuisine, wildlife, transportation, education, demographics, lesser-known places, or unique traditions.
- For CATEGORY slots: vary the specific sub-topic within the category. If Science, don't always ask about physics — try chemistry, biology, astronomy, earth science, etc. If History, don't always ask about the same era or country.
- Do NOT reuse the same core fact with different wording or options.
- Do NOT change only the options while keeping the same question concept.
- Each question must have exactly 4 text options and exactly 1 correct answer.
- The answer must exactly match one of the 4 options (verbatim).
- All questions must be factually accurate.
{existing_block}
{rejected_block}
Respond with a JSON array. Each element:
{{
  "slot_type": "city" or "category",
  "slot_value": "the city name or category name",
  "question": "...",
  "options": ["A", "B", "C", "D"],
  "answer": "one of the options"
}}

Return ONLY the JSON array, nothing else."""


# ── This or That prompt ─────────────────────────────────────────────

def _build_this_that_prompt(
    slots: list[tuple[str, str]],
    existing_questions: list[str],
    rejected_topics: dict[tuple[str, str], list[str]] | None = None,
) -> str:
    existing_block = ""
    if existing_questions:
        sample = existing_questions[:80]
        existing_block = (
            "\n\nEXISTING QUESTIONS (do NOT repeat, paraphrase, reword, or use the same underlying fact):\n"
            + "\n".join(f"- {q}" for q in sample)
        )

    rejected_block = _rejected_block(rejected_topics or {}, slots)

    return f"""Generate "This or That" questions for the following slots.

SLOTS TO FILL:
{_slots_description(slots)}

CRITICAL DIVERSITY RULES:
- Generate EXACTLY one question per slot.
- Every question must test a GENUINELY DIFFERENT fact from any existing question.
- For CITY slots: use lesser-known facts. Explore culture, festivals, cuisine, wildlife, transport, education, demographics, traditions.
- For CATEGORY slots: vary the specific sub-topic within the category.
- Do NOT reuse the same core fact with different wording.
- Each question has exactly 2 text options and 1 correct answer.
- The answer must exactly match one of the 2 options.
- All questions must be factually accurate.
{existing_block}
{rejected_block}
Respond with a JSON array. Each element:
{{
  "slot_type": "city" or "category",
  "slot_value": "the city name or category name",
  "question": "...",
  "option_a": "...",
  "option_b": "...",
  "answer": "must be option_a or option_b"
}}

Return ONLY the JSON array, nothing else."""


# ── Guess the Number prompt ─────────────────────────────────────────

def _build_guess_number_prompt(
    slots: list[tuple[str, str]],
    existing_questions: list[str],
    rejected_topics: dict[tuple[str, str], list[str]] | None = None,
) -> str:
    existing_block = ""
    if existing_questions:
        sample = existing_questions[:80]
        existing_block = (
            "\n\nEXISTING QUESTIONS (do NOT repeat, paraphrase, reword, or use the same underlying fact):\n"
            + "\n".join(f"- {q}" for q in sample)
        )

    rejected_block = _rejected_block(rejected_topics or {}, slots)

    return f"""Generate "Guess the Number" questions for the following slots.

SLOTS TO FILL:
{_slots_description(slots)}

CRITICAL DIVERSITY RULES:
- Generate EXACTLY one question per slot.
- Every question must test a GENUINELY DIFFERENT numerical fact.
- For CITY slots: use diverse numerical facts — population, area, founding year, number of schools, distance to another city, elevation, number of wards, etc.
- For CATEGORY slots: vary the specific sub-topic.
- Do NOT reuse the same core fact with different wording.
- Each question asks about a number, year, count, distance, or quantity.
- Each question has exactly 4 options (as strings) and 1 correct answer.
- The answer must exactly match one of the 4 options.
- All questions must be factually accurate.
{existing_block}
{rejected_block}
Respond with a JSON array. Each element:
{{
  "slot_type": "city" or "category",
  "slot_value": "the city name or category name",
  "question": "...",
  "options": ["100", "200", "300", "400"],
  "answer": "one of the options"
}}

Return ONLY the JSON array, nothing else."""


# ── Wrong Answer Only prompt ────────────────────────────────────────

def _build_wrong_answer_prompt(
    slots: list[tuple[str, str]],
    existing_questions: list[str],
    rejected_topics: dict[tuple[str, str], list[str]] | None = None,
) -> str:
    existing_block = ""
    if existing_questions:
        sample = existing_questions[:80]
        existing_block = (
            "\n\nEXISTING QUESTIONS (do NOT repeat, paraphrase, reword, or use the same underlying fact):\n"
            + "\n".join(f"- {q}" for q in sample)
        )

    rejected_block = _rejected_block(rejected_topics or {}, slots)

    return f"""Generate "Wrong Answer Only" questions for the following slots.

SLOTS TO FILL:
{_slots_description(slots)}

CRITICAL DIVERSITY RULES:
- Generate EXACTLY one question per slot.
- Every question must test a GENUINELY DIFFERENT fact from any existing question.
- For CITY slots: use lesser-known facts. Explore culture, festivals, cuisine, wildlife, transport, education, demographics, traditions.
- For CATEGORY slots: vary the specific sub-topic within the category.
- Do NOT reuse the same core fact with different wording.
- Each question has exactly 2 items: one correct option and one plausible wrong option.
- IMPORTANT: The "answer" field MUST be the WRONG option. This is the game mechanic.
- Both options should sound plausible so the game is fun.
- All questions must be factually accurate.
{existing_block}
{rejected_block}
Respond with a JSON array. Each element:
{{
  "slot_type": "city" or "category",
  "slot_value": "the city name or category name",
  "question": "...",
  "correct_option": "the factually correct answer",
  "wrong_option": "a plausible-sounding wrong answer",
  "answer": "MUST equal wrong_option"
}}

Return ONLY the JSON array, nothing else."""


_PROMPT_BUILDERS = {
    "quiz":            _build_quiz_prompt,
    "this-that":       _build_this_that_prompt,
    "guessthenumber":  _build_guess_number_prompt,
    "wrongansweronly":  _build_wrong_answer_prompt,
}


def generate_batch(
    game: str,
    slots: list[tuple[str, str]],
    existing_questions: list[str] | None = None,
    rejected_topics: dict[tuple[str, str], list[str]] | None = None,
    model_name: str = "gemini-3.6-flash",
) -> list[dict] | None:
    """
    Generate questions for all given slots in ONE LLM call.

    Args:
        game:               Game type key.
        slots:              List of (slot_type, slot_value) tuples to fill.
        existing_questions: Question texts already in the DB (for dedup guidance).
        rejected_topics:    Map of slot → list of rejected question texts.
        model_name:         Gemini model to use.

    Returns:
        Parsed list of dicts from Gemini, or None on failure.
        Returns "503" (string) on transient server errors for backoff handling.
    """
    builder = _PROMPT_BUILDERS.get(game)
    if not builder:
        raise ValueError(f"Unknown game type: {game}")

    prompt = builder(slots, existing_questions or [], rejected_topics)
    client = _get_client()

    try:
        response = client.models.generate_content(
            model=model_name,
            contents=prompt,
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
                temperature=1.0,
                automatic_function_calling=types.AutomaticFunctionCallingConfig(
                    disable=True,
                ),
            ),
        )
        raw = response.text.strip()
        parsed = json.loads(raw)

        if isinstance(parsed, dict):
            parsed = [parsed]

        if not isinstance(parsed, list):
            print(f"    ✗ Gemini returned unexpected type: {type(parsed)}")
            return None

        return parsed

    except json.JSONDecodeError as e:
        print(f"    ✗ JSON parse error: {e}")
        return None
    except (TimeoutError, ConnectionError, OSError) as e:
        print(f"    ✗ Timeout/connection error: {str(e)[:100]}")
        return "503"  # treat as transient
    except Exception as e:
        err_str = str(e)
        if any(k in err_str for k in ("503", "UNAVAILABLE", "timeout", "Timeout", "ReadTimeout")):
            print(f"    ✗ Gemini 503/timeout: {err_str[:120]}")
            return "503"
        if "429" in err_str or "RESOURCE_EXHAUSTED" in err_str:
            print(f"    ✗ Gemini 429 (rate limit): {err_str[:120]}")
            return "503"
        print(f"    ✗ Gemini API error: {e}")
        return None

