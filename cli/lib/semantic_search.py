import os

import numpy as np
from lib.search_utils import MOVIE_EMBEDDINGS_PATH, load_movies
from sentence_transformers import SentenceTransformer


class SemanticSearch:
    def __init__(self) -> None:
        self.model = SentenceTransformer("all-MiniLM-L6-V2")
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
