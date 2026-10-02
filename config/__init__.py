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

# ── Fixed 17 slots per game (split into 2 parts: 10 cities + 7 categories) ────
CITY_SLOTS = [("city", c) for c in CITIES]
CATEGORY_SLOTS = [("category", c) for c in CATEGORIES]
REQUIRED_SLOTS = CITY_SLOTS + CATEGORY_SLOTS

# ── Embedding ───────────────────────────────────────────────────────
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
EMBEDDING_DIMENSION = 384
EMBEDDINGS_COLLECTION = "question-embeddings"

# ── Generation ──────────────────────────────────────────────────────
MAX_DUPLICATE_RETRIES = 8          # max LLM requests per game for retries
GEMINI_MODEL = "gemini-3.5-flash"  # primary model (verified active & fast)

# Fallback models tried in order when the primary model returns 503/errors
GEMINI_FALLBACK_MODELS = [
    "gemini-3.5-flash",
    "gemini-3.7-flash",
    "gemini-flash-latest",
    "gemini-3.5-flash-lite",
]

# Exponential backoff delays (seconds) — applied with ±30% jitter
BACKOFF_DELAYS = [10, 30, 60, 120]

# Verification: if 503 prevents Gemini verification, questions with
# vector-search score BELOW this threshold are accepted (likely unique).
# Questions at or above are rejected as likely duplicates.
VERIFICATION_503_SCORE_THRESHOLD = 0.93

# Number of retries for the duplicate-verification Gemini call
VERIFICATION_RETRIES = 2
VERIFICATION_RETRY_DELAY = 10      # seconds between verification retries

# Max parallel Tavily searches
TAVILY_MAX_WORKERS = 6
