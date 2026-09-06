"""
Tavily Search Service — researches fresh facts for question generation.

Uses Tavily Search API to find lesser-known, reliable facts for
each pending slot before passing them to Gemini for question creation.
"""

import os
import random
from typing import Optional

from tavily import TavilyClient


# Topic variety hints to avoid repetitive searches
_CITY_TOPICS = [
    "historical events",
    "notable people from",
    "local festivals and traditions in",
    "local cuisine and food specialties of",
    "wildlife and nature near",
    "educational institutions in",
    "architecture and monuments in",
    "local industries and economy of",
    "geography and rivers near",
    "transportation and infrastructure in",
    "cultural heritage of",
    "sports and recreation in",
    "lesser-known facts about",
    "recent developments in",
    "demographics and communities of",
    "temples and religious sites in",
    "art and craft traditions of",
    "local markets and trade in",
]

_CATEGORY_TOPICS = {
    "General Knowledge": [
        "unusual world records",
        "surprising scientific discoveries",
        "lesser-known inventions",
        "unusual geographic facts",
        "unique cultural traditions worldwide",
        "interesting economic facts",
    ],
    "Geography": [
        "unusual landforms",
        "lesser-known rivers and lakes",
        "unique climate phenomena",
        "remote islands and territories",
        "geographical boundaries and borders",
        "ocean floor features",
    ],
    "History": [
        "lesser-known ancient civilizations",
        "unusual historical events",
        "forgotten historical figures",
        "historical discoveries and excavations",
        "unusual wars and conflicts",
        "historical trade routes",
    ],
    "Science": [
        "unusual chemical elements properties",
        "lesser-known biological phenomena",
        "space exploration discoveries",
        "unusual physics concepts",
        "medical science breakthroughs",
        "environmental science facts",
    ],
    "Sports": [
        "unusual sports rules and records",
        "lesser-known Olympic events",
        "sports history milestones",
        "unusual sports venues",
        "lesser-known sports worldwide",
        "sports technology innovations",
    ],
    "TV": [
        "Indian television show trivia",
        "behind-the-scenes TV facts",
        "TV show records and milestones",
        "lesser-known TV show facts",
        "iconic TV moments",
        "TV industry innovations",
    ],
    "Technology": [
        "lesser-known tech inventions",
        "cybersecurity milestones",
        "unusual programming language facts",
        "AI and machine learning history",
        "internet history milestones",
        "space technology innovations",
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
) -> dict[tuple[str, str], str]:
    """
    Research facts for all pending slots using Tavily Search.

    Args:
        slots:              List of (slot_type, slot_value) to research.
        rejected_topics:    Previously rejected question texts per slot.
        used_search_topics: Track which topic hints were already used per slot.

    Returns:
        Dict mapping (slot_type, slot_value) → research context string.
        Slots that fail research will be absent from the result.
    """
    client = _get_client()
    results: dict[tuple[str, str], str] = {}
    if used_search_topics is None:
        used_search_topics = {}

    for slot_type, slot_value in slots:
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
