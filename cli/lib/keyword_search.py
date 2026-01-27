import string

from nltk.stem import PorterStemmer

from .search_utils import DEFAULT_SEARCH_LIMIT, load_movies, load_stop_word


def search_command(query: str, limit: int = DEFAULT_SEARCH_LIMIT) -> list[dict]:
    movies = load_movies()
    stopwords = load_stop_word()
    results = []
    for movie in movies:
        query_tokens = tokenize_text(query, stopwords)
        title_tokens = tokenize_text(movie["title"], stopwords)
        if has_matching_token(query_tokens, title_tokens):
            results.append(movie)
            if len(results) >= limit:
                break

    return results


def has_matching_token(query_tokens: list[str], title_tokens: list[str]) -> bool:
    stemmer = PorterStemmer()
    for query_token in query_tokens:
        for title_token in title_tokens:
            if stemmer.stem(query_token) in stemmer.stem(title_token):
                return True
    return False


def preprocess_text(text: str) -> str:
    text = text.lower()
    text = text.translate(str.maketrans("", "", string.punctuation))
    return text


def tokenize_text(text: str, stopwords: list[str]) -> list[str]:
    text = preprocess_text(text)
    tokens = text.split()
    valid_tokens = []
    for token in tokens:
        if token:
            if token in stopwords:
                continue
            valid_tokens.append(token)
    return valid_tokens
