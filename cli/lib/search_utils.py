import json
import os
from typing import Optional

from dotenv import load_dotenv
from google import genai

DEFAULT_SEARCH_LIMIT = 5
DEFAULT_CHUNK_SIZE = 200
MAX_CHUNK_SIZE = 4
DEFAULT_CHUNK_OVERLAP = 0
BM25_SEARCH_LIMIT = 5
DOCUMENT_PREVIEW_LENGTH = 100
SCORE_PRECISION = 3
DEFAULT_HYBRID_ALPHA = 0.5
DEFAULT_HYBRID_LIMIT = 5
DEFAULT_K = 60

BM25_K1 = 1.5
BM25_B = 0.75

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
DATA_PATH = os.path.join(PROJECT_ROOT, "data", "movies.json")
STOPWORDS_PATH = os.path.join(PROJECT_ROOT, "data", "stopwords.txt")
CACHE_DIR = os.path.join(PROJECT_ROOT, "cache")
MOVIE_EMBEDDINGS_PATH = os.path.join(PROJECT_ROOT, "cache/movie_embeddings.npy")
CHUNK_EMBEDDINGS_PATH = os.path.join(PROJECT_ROOT, "cache/chunk_embeddings.npy")
CHUNK_METADATA_PATH = os.path.join(PROJECT_ROOT, "cache/chunk_metadata.json")


# AI Model information and setup
load_dotenv()
api_key = os.environ.get("GEMINI_API_KEY")
if api_key != None:
    print(f"Using key {api_key[:6]}...")
    client = genai.Client(api_key=api_key)


def load_movies() -> list[dict]:
    with open(DATA_PATH, "r") as f:
        data = json.load(f)
    return data["movies"]


def load_stopwords() -> list[str]:
    with open(STOPWORDS_PATH, "r") as f:
        return f.read().splitlines()


def hybrid_score(
    bm25_score: float, semantic_score: float, alpha: float = DEFAULT_HYBRID_ALPHA
):
    return alpha * bm25_score + (1 - alpha) * semantic_score


def normalize_scores(scores: list[float]):
    if scores is None or len(scores) == 0:
        return []

    min_score = min(scores)
    max_score = max(scores)

    scores_normalized: list[float] = []

    if min_score == max_score:
        for score in scores:
            scores_normalized.append(1.0)
    else:
        for score in scores:
            current_score = (score - min_score) / (max_score - min_score)
            scores_normalized.append(current_score)
    return scores_normalized


def rrf_score(rank: int, k: int = DEFAULT_K):
    if rank is None:
        return 0
    return 1 / (k + rank)


def spell_correct(query: str) -> str:
    prompt = f"""Fix any spelling errors in this movie search query.

    Only correct obvious typos. Don't change correctly spelled words.

    Query: "{query}"

    If no errors, return the original query.
    Corrected:"""

    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=prompt,
    )
    corrected = (response.text or "").strip().strip('"')
    return corrected if corrected else query


def rewrite_query(query: str) -> str:
    prompt = f"""Rewrite this movie search query to be more specific and searchable.

    Original: "{query}"

    Consider:
    - Common movie knowledge (famous actors, popular films)
    - Genre conventions (horror = scary, animation = cartoon)
    - Keep it concise (under 10 words)
    - It should be a google style search query that's very specific
    - Don't use boolean logic

    Examples:

    - "that bear movie where leo gets attacked" -> "The Revenant Leonardo DiCaprio bear attack"
    - "movie about bear in london with marmalade" -> "Paddington London marmalade"
    - "scary movie with bear from few years ago" -> "bear horror movie 2015-2020"

    Rewritten query:"""

    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=prompt,
    )
    corrected = (response.text or "").strip().strip('"')
    return corrected if corrected else query


def enhance_query(query: str, method: Optional[str] = None) -> str:
    match method:
        case "spell":
            return spell_correct(query)
        case "rewrite":
            return rewrite_query(query)
        case _:
            return query
