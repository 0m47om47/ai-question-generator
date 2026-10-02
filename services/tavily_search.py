"""
Tavily Search Service — researches fresh facts for question generation.

Uses Tavily Search API to find lesser-known, reliable facts for
each pending slot before passing them to Gemini for question creation.
"""

import os
import random
from typing import Optional

from tavily import TavilyClient


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


def _get_client() -> TavilyClient:
    api_key = os.getenv("TAVILY_API_KEY")
    if not api_key:
        raise ValueError("TAVILY_API_KEY is not set in .env")
    return TavilyClient(api_key=api_key)


def _pick_topic_hint(slot_type: str, slot_value: str,
                      used_topics: list[str]) -> str:
    """Pick a random topic hint, avoiding recently used ones."""
    if slot_type == "city":
        pool = _CITY_TOPICS
    else:
        pool = _CATEGORY_TOPICS.get(slot_value, ["interesting facts about"])

    # Filter out recently used
    available = [t for t in pool if t not in used_topics]
    if not available:
        available = pool  # reset if all used
    return random.choice(available)


def research_slots(
    slots: list[tuple[str, str]],
    rejected_topics: dict[tuple[str, str], list[str]] | None = None,
    used_search_topics: dict[tuple[str, str], list[str]] | None = None,
    cities_only: bool = True,
) -> dict[tuple[str, str], str]:
    """
    Research facts for pending slots using Tavily Search.

    Args:
        slots:              List of (slot_type, slot_value) to research.
        rejected_topics:    Previously rejected question texts per slot.
        used_search_topics: Track which topic hints were already used per slot.
        cities_only:        If True (default), only research city slots and skip
                            standard categories (GK, Science, TV, etc.) where
                            Gemini already possesses comprehensive knowledge.

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

    client = _get_client()
    results: dict[tuple[str, str], str] = {}
    if used_search_topics is None:
        used_search_topics = {}

    for slot_type, slot_value in target_slots:
        # Pick a diverse topic angle
        used = used_search_topics.get((slot_type, slot_value), [])
        topic_hint = _pick_topic_hint(slot_type, slot_value, used)
        used_search_topics.setdefault((slot_type, slot_value), []).append(topic_hint)

        # Build search query
        if slot_type == "city":
            query = f"{topic_hint} {slot_value} India"
        else:
            query = f"{topic_hint} {slot_value}"

        # Add anti-repetition if we have rejected questions
        rejected = rejected_topics.get((slot_type, slot_value), []) if rejected_topics else []

        print(f"    🔎 Researching: {slot_value} ({topic_hint})")

        try:
            response = client.search(
                query=query,
                search_depth="basic",
                max_results=3,
                include_answer=True,
            )

            # Extract useful context
            context_parts = []

            # Tavily's generated answer (most useful)
            answer = response.get("answer")
            if answer:
                context_parts.append(answer)

            # Top result snippets
            for r in response.get("results", [])[:3]:
                content = r.get("content", "")
                if content and len(content) > 30:
                    # Trim to useful length
                    context_parts.append(content[:500])

            if context_parts:
                context = "\n".join(context_parts)
                # Trim total context to avoid huge prompts
                results[(slot_type, slot_value)] = context[:1500]
            else:
                print(f"      ⚠ No useful results for {slot_value}")

        except Exception as e:
            err_str = str(e)
            if "429" in err_str or "rate" in err_str.lower():
                print(f"      ✗ Tavily rate limit: {err_str[:80]}")
                # Stop further searches on rate limit
                break
            print(f"      ✗ Tavily error for {slot_value}: {err_str[:80]}")
            continue

    return results
