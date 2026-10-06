"""
Web Search Service — researches fresh facts for question generation.

Uses You.com Search API to find lesser-known, reliable facts for
each pending slot before passing them to Gemini for question creation.

Supports both regional city slots and standard category slots (GK, Science,
Sports, History, TV, Technology, Geography).
Uses ThreadPoolExecutor for concurrent search requests.
"""

import os
import random
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Optional

import requests

from config import YOU_MAX_WORKERS


# ── You.com Search API configuration ────────────────────────────────
_YOU_API_URL = "https://ydc-index.io/v1/search"
_YOU_TIMEOUT = 15  # seconds per request


# Topic variety hints for engaging, accessible quiz questions (not obscure census/academic trivia)
_CITY_TOPICS = [
    "famous tourist attractions and landmarks in",
    "famous local food, dishes and sweets of",
    "popular festivals and cultural celebrations in",
    "famous temples, monuments and heritage sites in",
    "popular sports, parks and recreational places in",
    "interesting stories, history and nicknames of",
    "famous handicrafts, arts and specialties of",
    "prominent personalities, leaders and artists from",
    "famous natural spots, hills, lakes and waterfalls near",
    "popular markets and shopping areas in",
    "fun and interesting facts about",
    "major educational institutions and universities in",
]

_CATEGORY_TOPICS = {
    "General Knowledge": [
        "famous world landmarks and wonders",
        "popular world records and achievements",
        "famous currencies and flags of countries",
        "interesting world traditions and customs",
        "famous awards and honors like Nobel or Oscars",
        "well-known global organizations and capitals",
    ],
    "Geography": [
        "famous mountain peaks, ranges and volcanoes",
        "well-known rivers, oceans and waterfalls",
        "world deserts, forests and national parks",
        "famous world capitals and major cities",
        "distinctive geographical landmarks and continents",
        "well-known islands, beaches and straits",
    ],
    "History": [
        "famous ancient monuments and civilizations",
        "major historical events and independence movements",
        "famous rulers, kings and world leaders",
        "historic battles and turning points",
        "historic explorers and inventions in history",
        "world heritage sites and architectural wonders",
    ],
    "Science": [
        "everyday science facts and common phenomena",
        "famous inventions and their inventors",
        "solar system, planets and space missions",
        "fascinating animal and plant facts",
        "human body and health basics",
        "famous scientists and their discoveries",
    ],
    "Sports": [
        "Cricket World Cup and IPL records and legends",
        "Olympic Games famous moments and medalists",
        "Football (soccer) legends and tournaments",
        "popular Indian sports personalities and trophies",
        "Badminton, Tennis and Grand Slam champions",
        "famous sports rules, terms and stadiums",
    ],
    "TV": [
        "iconic Indian TV serials and sitcoms",
        "popular reality shows and game shows",
        "famous cartoon shows and childhood classics",
        "popular TV actors and character names",
        "classic TV show catchphrases and theme songs",
        "award-winning TV shows and milestones",
    ],
    "Technology": [
        "popular social media apps and tech founders",
        "smartphone history and popular gadgets",
        "well-known computer terms and internet milestones",
        "famous video games and gaming consoles",
        "everyday AI and digital innovations",
        "pioneering tech companies and their products",
    ],
}


def _get_api_key() -> str:
    api_key = os.getenv("YDC_API_KEY")
    if not api_key:
        raise ValueError("YDC_API_KEY is not set in .env")
    return api_key


def _pick_topic_hint(slot_type: str, slot_value: str,
                      used_topics: list[str]) -> str:
    """Pick a random topic hint, avoiding recently used ones."""
    if slot_type == "city":
        pool = _CITY_TOPICS
    else:
        pool = _CATEGORY_TOPICS.get(slot_value, [f"interesting facts about {slot_value}"])

    # Filter out recently used
    available = [t for t in pool if t not in used_topics]
    if not available:
        available = pool  # reset if all used
    return random.choice(available)


def _you_search(api_key: str, query: str, count: int = 3) -> dict:
    """
    Call the You.com Web Search API.

    Args:
        api_key: Your You.com API key.
        query:   Search query string.
        count:   Max results per section (web, news). Default 3.

    Returns:
        Parsed JSON response dict.
    """
    headers = {
        "X-API-Key": api_key,
        "Content-Type": "application/json",
    }
    payload = {
        "query": query,
        "count": count,
    }
    resp = requests.post(
        _YOU_API_URL,
        headers=headers,
        json=payload,
        timeout=_YOU_TIMEOUT,
    )
    resp.raise_for_status()
    return resp.json()


def _research_single_slot(
    api_key: str,
    slot_type: str,
    slot_value: str,
    topic_hint: str,
) -> tuple[tuple[str, str], Optional[str]]:
    """Research a single slot using You.com and return ((slot_type, slot_value), context_or_None)."""
    if slot_type == "city":
        query = f"{topic_hint} {slot_value} India"
    elif slot_value in _CATEGORY_TOPICS:
        query = topic_hint
    else:
        query = f"{topic_hint} {slot_value}"

    try:
        response = _you_search(api_key, query, count=3)
        context_parts = []

        # Web results — description + snippets
        web_results = response.get("results", {}).get("web", [])
        for r in web_results[:3]:
            desc = r.get("description", "")
            if desc and len(desc) > 30:
                context_parts.append(desc[:500])

            snippets = r.get("snippets", [])
            for snippet in snippets[:2]:
                if snippet and len(snippet) > 30:
                    context_parts.append(snippet[:400])

        # News results (if present)
        news_results = response.get("results", {}).get("news", [])
        for n in news_results[:2]:
            desc = n.get("description", "")
            if desc and len(desc) > 30:
                context_parts.append(desc[:400])

        if context_parts:
            context = "\n".join(context_parts)[:1500]
            return ((slot_type, slot_value), context)
        else:
            print(f"      ⚠ No useful results for {slot_value}")
            return ((slot_type, slot_value), None)

    except requests.exceptions.HTTPError as e:
        status_code = e.response.status_code if e.response is not None else 0
        if status_code == 429:
            print(f"      ✗ You.com rate limit (429) for {slot_value}: {str(e)[:80]}")
        elif status_code in (401, 403):
            print(f"      ✗ You.com auth error ({status_code}): {str(e)[:80]}")
        else:
            print(f"      ✗ You.com HTTP error ({status_code}) for {slot_value}: {str(e)[:80]}")
        return ((slot_type, slot_value), None)
    except requests.exceptions.Timeout:
        print(f"      ✗ You.com timeout for {slot_value}")
        return ((slot_type, slot_value), None)
    except Exception as e:
        print(f"      ✗ You.com error for {slot_value}: {str(e)[:80]}")
        return ((slot_type, slot_value), None)


def research_slots(
    slots: list[tuple[str, str]],
    rejected_topics: dict[tuple[str, str], list[str]] | None = None,
    used_search_topics: dict[tuple[str, str], list[str]] | None = None,
    cities_only: bool = False,
) -> dict[tuple[str, str], str]:
    """
    Research facts for pending slots using You.com Web Search in parallel.

    Args:
        slots:              List of (slot_type, slot_value) to research.
        rejected_topics:    Previously rejected question texts per slot.
        used_search_topics: Track which topic hints were already used per slot.
        cities_only:        If True, only research city slots. Default False (research both cities and categories).

    Returns:
        Dict mapping (slot_type, slot_value) → research context string.
        Slots that fail research will be absent from the result.
    """
    target_slots = [
        (stype, sval) for stype, sval in slots
        if (not cities_only or stype == "city")
    ]
    if not target_slots:
        return {}

    api_key = _get_api_key()
    results: dict[tuple[str, str], str] = {}
    if used_search_topics is None:
        used_search_topics = {}

    slot_tasks = []
    for slot_type, slot_value in target_slots:
        # Pick a diverse topic angle
        used = used_search_topics.get((slot_type, slot_value), [])
        topic_hint = _pick_topic_hint(slot_type, slot_value, used)
        used_search_topics.setdefault((slot_type, slot_value), []).append(topic_hint)
        slot_tasks.append((slot_type, slot_value, topic_hint))
        print(f"    🔎 Researching: {slot_value} ({topic_hint})")

    workers = min(YOU_MAX_WORKERS, len(slot_tasks))
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = [
            executor.submit(_research_single_slot, api_key, stype, sval, thint)
            for stype, sval, thint in slot_tasks
        ]
        for future in as_completed(futures):
            try:
                slot_key, context = future.result()
                if context:
                    results[slot_key] = context
            except Exception as e:
                print(f"      ✗ Search worker error: {e}")

    return results
