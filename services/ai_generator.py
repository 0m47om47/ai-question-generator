"""
AI Question Generator — batch-generates questions for missing slots
using Gemini with structured JSON output.

Integrates web research context when available for more diverse,
factually grounded questions. ONE LLM call per game (ideal case).
Follow-up calls only for missing / rejected / duplicate slots.

Resilience features:
- Model fallback chain (tries alternate models on 503/404)
- Dedicated retry loop for verification calls
- _is_model_error() skips dead models instead of aborting
"""

import json
import os
import time
from typing import Optional

from google import genai
from google.genai import types

from config import (
    GEMINI_MODEL,
    GEMINI_FALLBACK_MODELS,
    VERIFICATION_RETRIES,
    VERIFICATION_RETRY_DELAY,
)


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


def _is_transient_error(err_str: str) -> bool:
    """Check if an error string indicates a transient (retryable) error."""
    return any(k in err_str for k in (
        "503", "UNAVAILABLE", "timeout", "Timeout", "ReadTimeout",
        "429", "RESOURCE_EXHAUSTED",
    ))


def _is_model_error(err_str: str) -> bool:
    """Check if the error is model-specific (404 not found, deprecated, etc.).
    These errors mean THIS model won't work but another model might."""
    return any(k in err_str for k in (
        "404", "NOT_FOUND", "no longer available", "deprecated",
        "not supported", "does not exist",
    ))


def _call_gemini(
    client: genai.Client,
    model_name: str,
    prompt: str,
    temperature: float = 1.0,
) -> tuple[list[dict] | None, str | None]:
    """
    Make a single Gemini call and parse the JSON response.

    Returns:
        (parsed_list, None)   on success
        (None, "503")         on transient error (retry / try next model)
        (None, "model_error") on model-specific error (skip to next model)
        (None, error_msg)     on permanent error (stop trying)
    """
    try:
        response = client.models.generate_content(
            model=model_name,
            contents=prompt,
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
                temperature=temperature,
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
            return None, f"Unexpected type: {type(parsed)}"

        return parsed, None

    except json.JSONDecodeError as e:
        return None, f"JSON parse error: {e}"
    except (TimeoutError, ConnectionError, OSError) as e:
        return None, "503"
    except Exception as e:
        err_str = str(e)
        if _is_transient_error(err_str):
            return None, "503"
        if _is_model_error(err_str):
            return None, "model_error"
        return None, f"API error: {e}"


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


def _research_block(research_context: dict[tuple[str, str], str],
                     slots: list[tuple[str, str]]) -> str:
    """Build a prompt section with web research context per slot."""
    lines = []
    for slot in slots:
        context = research_context.get(slot)
        if context:
            stype, svalue = slot
            # Trim each context to ~400 chars for prompt efficiency
            trimmed = context[:400]
            lines.append(f"\n  {stype}: {svalue} — WEB RESEARCH:")
            lines.append(f"    {trimmed}")
    if lines:
        return "\nWEB RESEARCH CONTEXT (use these facts to create questions):\n" + "\n".join(lines) + "\n"
    return ""


_DIFFICULTY_GUIDELINES = """
QUESTION DIFFICULTY & QUALITY (CRITICAL):
- Target Audience: General quiz players, students, and casual gaming enthusiasts.
- Difficulty Level: Moderate, engaging, and fun (NOT obscure, academic, or PhD-level).
- Language: Natural, conversational, and direct (1-2 sentences maximum).
- STRICTLY DO NOT ASK:
  * Obscure census statistics (population density numbers, demographic decimals, exact census percentages).
  * Minor administrative bodies, micro-regulations, or obscure technical codes.
  * Overly academic or dry textbook jargon (e.g. niche scientific cellular pathways, obscure treaties).
- FOCUS ON:
  * Famous and beloved landmarks, tourist spots, hills, lakes, and monuments.
  * Renowned local dishes, sweets, snacks, and culinary specialties.
  * Popular festivals, cultural celebrations, and folk traditions.
  * Inspiring historical figures, legends, freedom fighters, and famous achievements.
  * Popular sports moments, everyday science, iconic TV/cinema, and major tech innovations.
  * Engaging facts that players find enjoyable and rewarding to answer!
"""


# ── Quiz prompt ─────────────────────────────────────────────────────

def _build_quiz_prompt(
    slots: list[tuple[str, str]],
    existing_questions: list[str],
    rejected_topics: dict[tuple[str, str], list[str]] | None = None,
    research_context: dict[tuple[str, str], str] | None = None,
) -> str:
    existing_block = ""
    if existing_questions:
        sample = existing_questions[:80]
        existing_block = (
            "\n\nEXISTING QUESTIONS (do NOT repeat, paraphrase, reword, or use the same underlying fact):\n"
            + "\n".join(f"- {q}" for q in sample)
        )

    rejected_block = _rejected_block(rejected_topics or {}, slots)
    research_block = _research_block(research_context or {}, slots)

    return f"""Generate quiz questions for the following slots.

SLOTS TO FILL:
{_slots_description(slots)}
{research_block}
{_DIFFICULTY_GUIDELINES}
CRITICAL DIVERSITY RULES:
- Generate EXACTLY one question per slot.
- Every question must test a GENUINELY DIFFERENT fact, event, person, place, concept, or statistic from existing questions.
- For CITY slots: Focus on cultural heritage, famous temples/sites, local delicacies/sweets, notable geographic spots (lakes/hills), or celebrated local legends. Do NOT ask for census numbers or dry statistics.
- For CATEGORY slots: Vary the specific sub-topic. Keep questions interesting, relatable, and educational without being dry or obscure.
- If web research context is provided above, USE IT to create the question. Do not ignore the research.
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
    research_context: dict[tuple[str, str], str] | None = None,
) -> str:
    existing_block = ""
    if existing_questions:
        sample = existing_questions[:80]
        existing_block = (
            "\n\nEXISTING QUESTIONS (do NOT repeat, paraphrase, reword, or use the same underlying fact):\n"
            + "\n".join(f"- {q}" for q in sample)
        )

    rejected_block = _rejected_block(rejected_topics or {}, slots)
    research_block = _research_block(research_context or {}, slots)

    return f"""Generate "This or That" questions for the following slots.

SLOTS TO FILL:
{_slots_description(slots)}
{research_block}
{_DIFFICULTY_GUIDELINES}
CRITICAL DIVERSITY RULES:
- Generate EXACTLY one question per slot.
- Every question must test a GENUINELY DIFFERENT fact from any existing question.
- For CITY slots: Focus on cultural heritage, famous temples/sites, local delicacies/sweets, notable geographic spots (lakes/hills), or celebrated local legends. Do NOT ask for census numbers or dry statistics.
- For CATEGORY slots: vary the specific sub-topic within the category.
- If web research context is provided above, USE IT to create the question.
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
    research_context: dict[tuple[str, str], str] | None = None,
) -> str:
    existing_block = ""
    if existing_questions:
        sample = existing_questions[:80]
        existing_block = (
            "\n\nEXISTING QUESTIONS (do NOT repeat, paraphrase, reword, or use the same underlying fact):\n"
            + "\n".join(f"- {q}" for q in sample)
        )

    rejected_block = _rejected_block(rejected_topics or {}, slots)
    research_block = _research_block(research_context or {}, slots)

    return f"""Generate "Guess the Number" questions for the following slots.

SLOTS TO FILL:
{_slots_description(slots)}
{research_block}
{_DIFFICULTY_GUIDELINES}
CRITICAL DIVERSITY RULES:
- Generate EXACTLY one question per slot.
- Every question must test a GENUINELY DIFFERENT numerical fact.
- For CITY slots: use engaging and recognizable numbers — founding or historical year, elevation in meters/feet, distance to a major city in km, number of major hills/lakes/platforms. NEVER ask for obscure census decimals or population densities.
- For CATEGORY slots: vary the specific sub-topic. Use memorable numbers (e.g. sports championship years, space mission years, world record counts).
- If web research context is provided above, USE IT to create the question.
- Do NOT reuse the same core fact with different wording.
- Each question asks about a number, year, count, distance, or quantity.
- Each question has exactly 4 options (as strings) and 1 correct answer.
- The answer must exactly match one of the 4 options.
- All questions must be factually accurate.

⚠️ ABSOLUTE RULE — NUMERIC OPTIONS ONLY:
- EVERY option MUST be a pure number (digits only). Examples: "1947", "342", "8848", "12".
- Options must NEVER contain words, letters, phrases, units, or text of any kind.
- WRONG examples: "Mount Everest", "New Delhi", "About 500", "~200", "500 km", "Three".
- RIGHT examples: "1947", "500", "8848", "12", "2023", "365".
- The "answer" field must also be a pure number matching one of the 4 numeric options.
- If you generate ANY option that is not purely digits, the response will be REJECTED.
{existing_block}
{rejected_block}
Respond with a JSON array. Each element:
{{
  "slot_type": "city" or "category",
  "slot_value": "the city name or category name",
  "question": "In which year was the Taj Mahal completed?",
  "options": ["1648", "1653", "1670", "1632"],
  "answer": "1653"
}}

Return ONLY the JSON array, nothing else."""


# ── Wrong Answer Only prompt ────────────────────────────────────────

def _build_wrong_answer_prompt(
    slots: list[tuple[str, str]],
    existing_questions: list[str],
    rejected_topics: dict[tuple[str, str], list[str]] | None = None,
    research_context: dict[tuple[str, str], str] | None = None,
) -> str:
    existing_block = ""
    if existing_questions:
        sample = existing_questions[:80]
        existing_block = (
            "\n\nEXISTING QUESTIONS (do NOT repeat, paraphrase, reword, or use the same underlying fact):\n"
            + "\n".join(f"- {q}" for q in sample)
        )

    rejected_block = _rejected_block(rejected_topics or {}, slots)
    research_block = _research_block(research_context or {}, slots)

    return f"""Generate "Wrong Answer Only" questions for the following slots.

SLOTS TO FILL:
{_slots_description(slots)}
{research_block}
{_DIFFICULTY_GUIDELINES}
CRITICAL DIVERSITY RULES:
- Generate EXACTLY one question per slot.
- Every question must test a GENUINELY DIFFERENT fact from any existing question.
- For CITY slots: Focus on cultural heritage, famous temples/sites, local delicacies/sweets, notable geographic spots (lakes/hills), or celebrated local legends. Do NOT ask for census numbers or dry statistics.
- For CATEGORY slots: vary the specific sub-topic within the category.
- If web research context is provided above, USE IT to create the question.
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
    research_context: dict[tuple[str, str], str] | None = None,
    model_name: str = GEMINI_MODEL,
) -> list[dict] | None:
    """
    Generate questions for all given slots in ONE LLM call.

    Uses a model fallback chain: if the primary model returns 503 or 404,
    tries alternate models before giving up.

    Returns:
        Parsed list of dicts from Gemini, or None on failure.
        Returns "503" (string) on transient server errors for backoff handling.
    """
    builder = _PROMPT_BUILDERS.get(game)
    if not builder:
        raise ValueError(f"Unknown game type: {game}")

    prompt = builder(slots, existing_questions or [], rejected_topics, research_context)
    client = _get_client()

    # Build the model chain: primary model first, then fallbacks (deduped)
    models_to_try = [model_name]
    for m in GEMINI_FALLBACK_MODELS:
        if m not in models_to_try:
            models_to_try.append(m)

    last_error = None
    for model in models_to_try:
        result, error = _call_gemini(client, model, prompt, temperature=1.0)

        if result is not None:
            if model != model_name:
                print(f"    ℹ Succeeded with fallback model: {model}")
            return result

        if error == "503":
            print(f"    ✗ {model} → 503/overloaded, trying next …")
            last_error = "503"
            continue
        elif error == "model_error":
            print(f"    ✗ {model} → model unavailable, trying next …")
            last_error = last_error or "model_error"
            continue
        else:
            # Truly permanent error (e.g. bad JSON from a successful response)
            print(f"    ✗ {model} → {error}")
            return None

    # All models exhausted
    if last_error == "503":
        print(f"    ✗ All {len(models_to_try)} models returned 503")
        return "503"
    print(f"    ✗ All {len(models_to_try)} models failed")
    return None


def verify_duplicates_batch(
    candidates: list[dict],
    model_name: str = GEMINI_MODEL,
) -> dict[int, bool] | None:
    """
    Batch-verify potential duplicates with ONE Gemini call.

    Includes:
    - Model fallback chain (tries alternate models on 503/404)
    - Dedicated retry loop with backoff before giving up

    Returns:
        Dict mapping candidate index → True (duplicate) / False (unique).
        Returns None on permanent failure, or "503" on transient failure.
    """
    if not candidates:
        return {}

    # ── Build prompt ────────────────────────────────────────────────
    pair_lines = []
    for i, c in enumerate(candidates):
        pair_lines.append(
            f"  {i+1}. NEW: \"{c['new_question']}\"\n"
            f"     EXISTING: \"{c['existing_question']}\"\n"
            f"     SIMILARITY: {c['score']:.3f}"
        )
    pairs_block = "\n".join(pair_lines)

    prompt = f"""You are a duplicate-detection judge for quiz questions.

For each numbered pair below, decide whether the NEW question tests the
**same underlying fact** as the EXISTING question.

- If the core fact, answer, and intent are the same (even if worded differently),
  respond "DUPLICATE".
- If they test genuinely different facts, topics, or aspects — even if they
  share a city name, category, or some vocabulary — respond "UNIQUE".

PAIRS TO EVALUATE:
{pairs_block}

Respond with a JSON array of objects, one per pair, in order:
[
  {{"pair": 1, "verdict": "DUPLICATE" or "UNIQUE"}},
  {{"pair": 2, "verdict": "DUPLICATE" or "UNIQUE"}},
  ...
]

Return ONLY the JSON array, nothing else."""

    client = _get_client()

    # Build the model chain
    models_to_try = [model_name]
    for m in GEMINI_FALLBACK_MODELS:
        if m not in models_to_try:
            models_to_try.append(m)

    # Retry loop: try each model, then retry with backoff
    for retry in range(VERIFICATION_RETRIES + 1):
        for model in models_to_try:
            result, error = _call_gemini(client, model, prompt, temperature=0.0)

            if result is not None:
                if model != model_name:
                    print(f"    ℹ Verification succeeded with fallback: {model}")
                return _parse_verification_verdicts(result, candidates)

            if error == "503":
                print(f"    ✗ Verification: {model} → 503/overloaded")
                continue
            elif error == "model_error":
                print(f"    ✗ Verification: {model} → model unavailable")
                continue
            else:
                print(f"    ✗ Verification: {model} → {error}")
                return None

        # All models failed this round — backoff before next retry
        if retry < VERIFICATION_RETRIES:
            print(f"    ⏳ Verification retry {retry+1}/{VERIFICATION_RETRIES} in {VERIFICATION_RETRY_DELAY}s …")
            time.sleep(VERIFICATION_RETRY_DELAY)

    # All retries exhausted
    print(f"    ✗ Verification failed after {VERIFICATION_RETRIES + 1} rounds × {len(models_to_try)} models")
    return "503"


def _parse_verification_verdicts(
    parsed: list[dict],
    candidates: list[dict],
) -> dict[int, bool]:
    """Parse Gemini's verification response into candidate index → is_duplicate map."""
    results: dict[int, bool] = {}
    for entry in parsed:
        pair_num = entry.get("pair", 0)
        verdict = str(entry.get("verdict", "")).upper().strip()
        if 1 <= pair_num <= len(candidates):
            candidate_idx = candidates[pair_num - 1]["index"]
            results[candidate_idx] = (verdict == "DUPLICATE")

    # Fill in any missing candidates as DUPLICATE (safe default)
    for c in candidates:
        if c["index"] not in results:
            results[c["index"]] = True

    return results
