# AI Question Generator — Phase 1

Embeds all existing quiz questions from MongoDB into a new `question-embeddings` collection using a free, local embedding model.

## Prerequisites

- Python 3.9+
- Access to the existing `quizappdb` MongoDB database

## Setup

```bash
cd ai-question-generator

# Create virtual environment
python -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

## Configuration

Copy the example env file and fill in your MongoDB URI:

```bash
cp .env.example .env
```

Edit `.env`:

```
MONGODB_URI=mongodb+srv://<user>:<pass>@<cluster>.mongodb.net/?retryWrites=true&w=majority
MONGODB_DATABASE=quizappdb
```

## Run Migration

```bash
python scripts/embed_existing_questions.py
```

**Safe to run multiple times** — duplicates are automatically detected and skipped.

### What it does

1. Connects to MongoDB (`quizappdb`)
2. Reads all documents from:
   - `quiz-game`
   - `this-that-game`
   - `guessthenumber-game`
   - `wrongansweronly-game`
3. Extracts each individual question
4. Generates a 384-dimensional embedding using `all-MiniLM-L6-v2` (runs locally, no API key needed)
5. Inserts embedding documents into `question-embeddings`
6. **Never modifies** the original game collections

### Output

```
Total questions found : N
Already embedded      : N
Newly embedded        : N
Failed                : N
```

## Test Embeddings

```bash
python scripts/test_embedding.py
```

Verifies that semantically similar questions score higher than unrelated ones.

## Embedding Document Schema

```json
{
  "game": "quiz",
  "type": "city",
  "city": "Dhanbad",
  "category": null,
  "question": "Which major coal subsidiary is headquartered in Dhanbad?",
  "options": ["BCCL", "ECL", "CCL", "WCL"],
  "answer": "BCCL",
  "embedding": [384-dim vector],
  "source": "existing",
  "sourceCollection": "quiz-game",
  "sourceDocumentId": "...",
  "sourceQuestionId": 1,
  "createdAt": "2026-08-09T..."
}
```

## Collections (unchanged)

```
quizappdb
├── quiz-game              ← unchanged
├── this-that-game         ← unchanged
├── guessthenumber-game    ← unchanged
├── wrongansweronly-game    ← unchanged
└── question-embeddings    ← NEW (Phase 1)
```
