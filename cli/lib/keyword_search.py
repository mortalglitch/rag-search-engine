import math
import os
import pickle
import string
from collections import Counter, defaultdict

from nltk.stem import PorterStemmer

from .search_utils import (
    BM25_K1,
    CACHE_DIR,
    DEFAULT_SEARCH_LIMIT,
    load_movies,
    load_stopwords,
)


class InvertedIndex:
    def __init__(self) -> None:
        self.index = defaultdict(set)
        self.docmap: dict[int, dict] = {}
        self.index_path = os.path.join(CACHE_DIR, "index.pkl")
        self.docmap_path = os.path.join(CACHE_DIR, "docmap.pkl")
        self.term_frequencies_path = os.path.join(CACHE_DIR, "term_frequencies.pkl")
        self.term_frequencies: dict[int, Counter] = {}

    def build(self) -> None:
        movies = load_movies()
        for m in movies:
            doc_id = m["id"]
            doc_description = f"{m['title']} {m['description']}"
            self.docmap[doc_id] = m
            self.__add_document(doc_id, doc_description)

    def save(self) -> None:
        os.makedirs(CACHE_DIR, exist_ok=True)
        with open(self.index_path, "wb") as f:
            pickle.dump(self.index, f)
        with open(self.docmap_path, "wb") as f:
            pickle.dump(self.docmap, f)
        with open(self.term_frequencies_path, "wb") as f:
            pickle.dump(self.term_frequencies, f)

    def get_bm25_idf(self, term: str) -> float:
        tokens = tokenize_text(term)

        if len(tokens) != 1:
            raise ValueError("IDF Search term must be one word.")

        total_doc_count = len(self.docmap)
        if tokens[0] in self.index:
            term_match_doc_count = len(self.index[tokens[0]])
        else:
            term_match_doc_count = 0

        return math.log(
            (total_doc_count - term_match_doc_count + 0.5)
            / (term_match_doc_count + 0.5)
            + 1
        )

    def get_bm25_tf(self, doc_id, term, k1=BM25_K1) -> float:
        raw_tf = self.get_tf(doc_id, term)
        bm25_tf = (raw_tf * (k1 + 1)) / (raw_tf + k1)
        return bm25_tf

    def get_documents(self, term: str) -> list[int]:
        doc_ids = self.index.get(term, set())
        return sorted(list(doc_ids))

    def get_idf(self, term: str) -> float:
        tokens = tokenize_text(term)

        if len(tokens) != 1:
            raise ValueError("IDF Search term must be one word.")

        total_doc_count = len(self.docmap)
        if tokens[0] in self.index:
            term_match_doc_count = len(self.index[tokens[0]])
        else:
            term_match_doc_count = 0

        return math.log((total_doc_count + 1) / (term_match_doc_count + 1))

    def get_tf(self, doc_id, term) -> int:
        term_tokens = tokenize_text(term)
        if len(term_tokens) > 1:
            raise ValueError("Only one term was expected")
        elif len(term_tokens) == 0:
            return 0
        else:
            return self.term_frequencies[doc_id][term_tokens[0]]
        return 0

    def get_tf_idf(self, doc_id: int, term: str) -> float:
        tf = self.get_tf(doc_id, term)
        idf = self.get_idf(term)

        return tf * idf

    def __add_document(self, doc_id: int, text: str) -> None:
        tokens = tokenize_text(text)

        if doc_id not in self.term_frequencies:
            self.term_frequencies[doc_id] = Counter()

        for token in set(tokens):
            self.index[token].add(doc_id)
        self.term_frequencies[doc_id].update(tokens)

    def load(self):
        try:
            with open(self.index_path, "rb") as f:
                self.index = pickle.load(f)
        except FileNotFoundError:
            print("Error: The index file does not exist.")
        try:
            with open(self.docmap_path, "rb") as f:
                self.docmap = pickle.load(f)
        except FileNotFoundError:
            print("The docmap file does not exist.")
        try:
            with open(self.term_frequencies_path, "rb") as f:
                self.term_frequencies = pickle.load(f)
        except FileNotFoundError:
            print("The term frequency file does not exist.")


def build_command() -> None:
    idx = InvertedIndex()
    idx.build()
    idx.save()

    # # Quick test for debugging
    # docs = idx.get_documents("merida")
    # print(f"First document for token 'merida' = {docs[0]}")


def bm25_idf_command(term: str):
    invertIndex = InvertedIndex()
    invertIndex.load()

    bm25idf = invertIndex.get_bm25_idf(term)
    return bm25idf


def bm25_tf_command(doc_id: str, term: str, k1=BM25_K1):
    invertIndex = InvertedIndex()
    invertIndex.load()

    bm25_tf = invertIndex.get_bm25_tf(doc_id, term, k1)
    return bm25_tf


def idf_command(term: str):
    invertIndex = InvertedIndex()
    invertIndex.load()

    idf = invertIndex.get_idf(term)
    return idf


def search_command(query: str, limit: int = DEFAULT_SEARCH_LIMIT) -> list[dict]:
    # Load Index
    invertIndex = InvertedIndex()
    invertIndex.load()

    query_tokens = tokenize_text(query)

    results = []

    for token in query_tokens:
        if len(results) >= limit:
            break
        current_movies = invertIndex.get_documents(token)
        for movie_id in current_movies:
            if len(results) >= limit:
                break
            results.append(invertIndex.docmap[movie_id])

    return results

    # movies = load_movies()
    # results = []
    # for movie in movies:
    #     query_tokens = tokenize_text(query)
    #     title_tokens = tokenize_text(movie["title"])
    #     if has_matching_token(query_tokens, title_tokens):
    #         results.append(movie)
    #         if len(results) >= limit:
    #             break

    return results


def tf_command(doc_id: int, term: str) -> int:
    invertIndex = InvertedIndex()
    invertIndex.load()

    return invertIndex.get_tf(doc_id, term)


def tf_idf_command(doc_id: int, term: str) -> float:
    invertIndex = InvertedIndex()
    invertIndex.load()

    return invertIndex.get_tf_idf(doc_id, term)


def has_matching_token(query_tokens: list[str], title_tokens: list[str]) -> bool:
    for query_token in query_tokens:
        for title_token in title_tokens:
            if query_token in title_token:
                return True
    return False


def preprocess_text(text: str) -> str:
    text = text.lower()
    text = text.translate(str.maketrans("", "", string.punctuation))
    return text


def tokenize_text(text: str) -> list[str]:
    text = preprocess_text(text)
    tokens = text.split()
    valid_tokens = []
    for token in tokens:
        if token:
            valid_tokens.append(token)
    stop_words = load_stopwords()
    filtered_words = []
    for word in valid_tokens:
        if word not in stop_words:
            filtered_words.append(word)
    stemmer = PorterStemmer()
    stemmed_words = []
    for word in filtered_words:
        stemmed_words.append(stemmer.stem(word))
    return stemmed_words
