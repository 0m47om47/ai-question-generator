"""
Central constants for the AI Question Generator.

These MUST match the existing preference system in the Quiz App.
Do NOT add or remove items without updating the frontend.
"""

# ── Game types & collection mapping ─────────────────────────────────
GAMES = ["quiz", "this-that", "guessthenumber", "wrongansweronly"]

GAME_COLLECTION_MAP = {
    "quiz":            "quiz-game",
    "this-that":       "this-that-game",
    "guessthenumber":  "guessthenumber-game",
    "wrongansweronly":  "wrongansweronly-game",
}

# ── The 10 configured cities (city-type slots) ─────────────────────
CITIES = [
    "Thane",
    "Bhagalpur",
    "Deoghar",
    "Dhanbad",
    "Dumka",
    "Godda",
    "Jamtara",
    "Pakur",
    "Rampurhat",
    "Sahibganj",
]

# ── The 7 configured categories (category-type slots) ──────────────
CATEGORIES = [
    "General Knowledge",
    "Geography",
    "History",
    "Science",
    "Sports",
    "TV",
    "Technology",
]

# ── Fixed 17 slots per game ─────────────────────────────────────────
# 10 city slots + 7 category slots = 17 total
REQUIRED_SLOTS = (
    [("city", c) for c in CITIES]
    + [("category", c) for c in CATEGORIES]
)

# ── Embedding ───────────────────────────────────────────────────────
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
EMBEDDING_DIMENSION = 384
EMBEDDINGS_COLLECTION = "question-embeddings"

# ── Generation ──────────────────────────────────────────────────────
MAX_DUPLICATE_RETRIES = 5          # max LLM requests per game for retries
GEMINI_MODEL = "gemini-3.6-flash"
