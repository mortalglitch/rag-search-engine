import json
import os
import re
from typing import Any

import numpy as np
from lib.search_utils import (
    CHUNK_EMBEDDINGS_PATH,
    CHUNK_METADATA_PATH,
    DEFAULT_CHUNK_OVERLAP,
    DEFAULT_CHUNK_SIZE,
    DEFAULT_SEARCH_LIMIT,
    MAX_CHUNK_SIZE,
    MOVIE_EMBEDDINGS_PATH,
    load_movies,
)
from sentence_transformers import SentenceTransformer


class SemanticSearch:
    def __init__(self, model_name="all-MiniLM-L6-v2") -> None:
        self.model = SentenceTransformer(model_name)
        self.embeddings = None
        self.documents = []
        self.document_map = {}

    def build_embeddings(self, documents):
        self.documents = documents

        doc_list: list[str] = []
        for doc in self.documents:
            self.document_map[doc["id"]] = doc
            doc_list.append(f"{doc['title']}: {doc['description']}")
        self.embeddings = self.model.encode(doc_list, show_progress_bar=True)

        with open(MOVIE_EMBEDDINGS_PATH, "wb") as f:
            np.save(f, self.embeddings)

        return self.embeddings

    def generate_embedding(self, text):
        if not text.strip():
            raise ValueError("Text must not contain whitespace")
        elif len(text) == 0:
            raise ValueError("Text must not be empty")

        text_store = []
        text_store.append(text)

        result = self.model.encode(text_store)

        return result[0]

    def load_or_create_embeddings(self, documents):
        self.documents = documents
        doc_list: list[str] = []
        for doc in self.documents:
            self.document_map[doc["id"]] = doc
            doc_list.append(f"{doc['title']}: {doc['description']}")

        if os.path.exists(MOVIE_EMBEDDINGS_PATH):
            self.embeddings = np.load(MOVIE_EMBEDDINGS_PATH)
            if (
                self.embeddings is not None
                and self.documents is not None
                and len(self.embeddings) == len(self.documents)
            ):
                return self.embeddings

        return self.build_embeddings(documents)

    def search(self, query, limit: int = DEFAULT_SEARCH_LIMIT):
        if self.embeddings is not None:
            embed_query = self.generate_embedding(query)
            results: list[tuple[float, dict[Any, Any]]] = []
            for index, embedding in enumerate(self.embeddings):
                similarity_score = cosine_similarity(embed_query, embedding)
                results.append((similarity_score, self.document_map[index + 1]))
            sorted_results = sorted(results, key=lambda item: item[0], reverse=True)

            limited_results = sorted_results[:limit]

            final_results: list[dict] = []
            for score, doc in limited_results:
                search_result: dict = {}
                search_result["id"] = doc["id"]
                search_result["title"] = doc["title"]
                search_result["description"] = doc["description"]
                search_result["score"] = score
                final_results.append(search_result)

            return final_results

        else:
            raise ValueError(
                "No embeddings loaded. Call `load_or_create_embeddings` first."
            )


class ChunkedSemanticSearch(SemanticSearch):
    def __init__(self, model_name="all-MiniLM-L6-v2") -> None:
        super().__init__(model_name)
        self.chunk_embeddings = None
        self.chunk_metadata = None

    def build_check_embeddings(self, documents):
        self.documents = documents

        chunk_list = []
        chunk_meta: list[dict] = []
        for idx, doc in enumerate(self.documents):
            self.document_map[doc["id"]] = doc
            if doc["description"] == "":
                pass
            chunks = semantic_chunk_text(doc["description"], 4, 1)
            for chunk_idx, chunk in enumerate(chunks):
                chunk_list.append(chunk)
                chunk_meta.append(
                    {
                        "movie_idx": idx,
                        "chunk_idx": chunk_idx,
                        "total_chunks": len(chunks),
                    }
                )

        self.chunk_embeddings = self.model.encode(chunk_list, show_progress_bar=True)

        with open(CHUNK_EMBEDDINGS_PATH, "wb") as f:
            np.save(f, self.chunk_embeddings)

        os.makedirs(os.path.dirname(CHUNK_METADATA_PATH), exist_ok=True)
        with open(CHUNK_METADATA_PATH, "w") as file:
            json.dump(
                {"chunks": self.chunk_metadata, "total_chunks": len(chunk_list)},
                file,
                indent=2,
            )

        return self.chunk_embeddings

    def load_or_create_chunk_embeddings(self, documents: list[dict]) -> np.ndarray:
        self.documents = documents
        doc_list: list[str] = []
        for doc in self.documents:
            self.document_map[doc["id"]] = doc
            doc_list.append(f"{doc['title']}: {doc['description']}")

        if os.path.exists(CHUNK_EMBEDDINGS_PATH) and os.path.exists(
            CHUNK_METADATA_PATH
        ):
            self.chunk_embeddings = np.load(CHUNK_EMBEDDINGS_PATH)

            with open(CHUNK_METADATA_PATH, "r") as f:
                self.chunk_metadata = json.load(f)["chunks"]

            return self.chunk_embeddings

        return self.build_check_embeddings(documents)


def chunk_text(
    text: str,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    overlap: int = DEFAULT_CHUNK_OVERLAP,
):
    split_words = text.split()
    # return [
    #     split_words[i : i + chunk_size] for i in range(0, len(split_words), chunk_size)
    # ] # - old one liner

    i = 0
    chunks: list[list[str]] = []
    while i < len(split_words) and (i == 0 or (len(split_words) - i) > overlap):
        chunk = split_words[i : i + chunk_size]
        chunks.append(chunk)
        i += chunk_size - overlap

    return chunks


def cosine_similarity(vec1, vec2):
    dot_product = np.dot(vec1, vec2)
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)

    if norm1 == 0 or norm2 == 0:
        return 0.0

    return dot_product / (norm1 * norm2)


def embed_chunks():
    movies = load_movies()
    chunked_semantic_search = ChunkedSemanticSearch()
    embeddings = chunked_semantic_search.load_or_create_chunk_embeddings(movies)
    print(f"Generated {len(embeddings)} chunked embeddings")


def embed_query_text(query):
    sem_search = SemanticSearch()
    embedding = sem_search.generate_embedding(query)
    print(f"Query: {query}")
    print(f"First 5 dimensions: {embedding[:5]}")
    print(f"Shape: {embedding.shape}")


def embed_text(text):
    sem_search = SemanticSearch()
    embedded_result = sem_search.generate_embedding(text)

    print(f"Text: {text}")
    print(f"First 3 dimensions: {embedded_result[:3]}")
    print(f"Dimensions: {embedded_result.shape[0]}")


def semantic_chunk_text(
    text: str,
    max_chunk_size: int = MAX_CHUNK_SIZE,
    overlap: int = DEFAULT_CHUNK_OVERLAP,
):
    split_words = re.split(r"(?<=[.!?])\s+", text)
    # return [
    #     split_words[i : i + chunk_size] for i in range(0, len(split_words), chunk_size)
    # ] # - old one liner

    i = 0
    chunks: list[str] = []
    while i < len(split_words) and (i == 0 or (len(split_words) - i) > overlap):
        chunk = split_words[i : i + max_chunk_size]
        chunks.append(" ".join(chunk))
        i += max_chunk_size - overlap

    return chunks


def semantic_search(query, limit):
    sem_search = SemanticSearch()
    documents = load_movies()
    sem_search.load_or_create_embeddings(documents)

    result = sem_search.search(query, limit)

    print(f"Query: {query}")
    print(f"Top {len(result)}:")

    for index, search_result in enumerate(result):
        print(
            f"{index + 1}. {search_result['title']} (Score: {search_result['score']:.4f})\n{search_result['description']}\n"
        )


def verify_embeddings():
    sem_search = SemanticSearch()
    documents = load_movies()
    embeddings = sem_search.load_or_create_embeddings(documents)

    print(f"Number of docs:   {len(documents)}")
    print(
        f"Embeddings shape: {embeddings.shape[0]} vectors in {embeddings.shape[1]} dimensions"
    )


def verify_model():
    new_semantic = SemanticSearch()
    print(f"Model loaded:{new_semantic.model}")
    print(f"Max sequence length: {new_semantic.model.max_seq_length}")
    return
