import os
from typing import Optional

from lib.search_utils import (
    DEFAULT_HYBRID_ALPHA,
    DEFAULT_HYBRID_LIMIT,
    DOCUMENT_PREVIEW_LENGTH,
    SCORE_PRECISION,
    enhance_query,
    hybrid_score,
    load_movies,
    normalize_scores,
    rrf_score,
)

from .keyword_search import InvertedIndex
from .semantic_search import ChunkedSemanticSearch


class HybridSearch:
    def __init__(self, documents):
        self.documents = documents
        self.semantic_search = ChunkedSemanticSearch()
        self.semantic_search.load_or_create_chunk_embeddings(documents)

        self.idx = InvertedIndex()
        if not os.path.exists(self.idx.index_path):
            self.idx.build()
            self.idx.save()

    def _bm25_search(self, query, limit):
        self.idx.load()
        return self.idx.bm25_search(query, limit)

    def weighted_search(self, query, alpha, limit):
        bm25_results = self._bm25_search(query, limit * 500)
        chunked_sem_results = self.semantic_search.search_chunks(query, limit * 500)

        bm25_scores = list(map(lambda result: result["score"], bm25_results))
        chunked_sem_scores = list(
            map(lambda result: result["score"], chunked_sem_results)
        )

        bm25_doc_ids = list(map(lambda result: result["id"], bm25_results))
        sem_doc_ids = list(map(lambda result: result["id"], chunked_sem_results))

        norm_bm25 = normalize_scores(bm25_scores)
        norm_chunked_sem = normalize_scores(chunked_sem_scores)

        norm_bm25_map = dict(zip(bm25_doc_ids, norm_bm25))
        norm_chunked_map = dict(zip(sem_doc_ids, norm_chunked_sem))

        norm_score_map: dict[int, dict] = {}
        for doc in self.documents:
            doc_id = doc["id"]
            norm_score_map[doc_id] = {}

            bm25_score = 0.0
            sem_score = 0.0

            if doc_id in norm_bm25_map:
                bm25_score = norm_bm25_map[doc_id]
            if doc_id in norm_chunked_map:
                sem_score = norm_chunked_map[doc_id]

            norm_score_map[doc_id]["doc"] = doc
            norm_score_map[doc_id]["norm_bm25_score"] = bm25_score
            norm_score_map[doc_id]["norm_semantic_score"] = sem_score
            norm_score_map[doc_id]["hybrid_score"] = hybrid_score(
                bm25_score,
                sem_score,
                alpha,
            )

        scores_sorted = sorted(
            norm_score_map.items(),
            key=lambda item: item[1]["hybrid_score"],
            reverse=True,
        )

        top_scores = scores_sorted[:limit]

        results: list[dict] = []
        for doc_id, result in top_scores:
            doc = result["doc"]
            search_result: dict = {}
            search_result["id"] = doc_id
            search_result["title"] = doc["title"]
            search_result["description"] = doc["description"][:DOCUMENT_PREVIEW_LENGTH]
            search_result["bm25_score"] = round(
                result["norm_bm25_score"], SCORE_PRECISION
            )
            search_result["semantic_score"] = round(
                result["norm_semantic_score"], SCORE_PRECISION
            )
            search_result["hybrid_score"] = round(
                result["hybrid_score"], SCORE_PRECISION
            )
            results.append(search_result)

        return results

    def rrf_search(self, query, k_value, limit):
        bm25_results = self._bm25_search(query, limit * 500)
        chunked_sem_results = self.semantic_search.search_chunks(query, limit * 500)

        bm25_with_id = list(map(lambda result: result["id"], bm25_results))
        chunked_with_id = list(map(lambda result: result["id"], chunked_sem_results))

        rrf_scores_map: dict[int, dict] = {}
        for doc in self.documents:
            doc_id = doc["id"]
            rrf_scores_map[doc_id] = {}

            bm25_rank: float = 0
            semantic_rank: float = 0

            if doc_id in bm25_with_id:
                bm25_rank = bm25_with_id.index(doc_id)
            if doc_id in chunked_with_id:
                semantic_rank = chunked_with_id.index(doc_id)

            rrf_scores_map[doc_id]["doc"] = doc
            rrf_scores_map[doc_id]["bm25_rank"] = bm25_rank
            rrf_scores_map[doc_id]["semantic_rank"] = semantic_rank
            bm25_rrf_score = rrf_score(bm25_rank, k_value)
            semantic_rrf_score = rrf_score(semantic_rank, k_value)
            rrf_scores_map[doc_id]["rrf_score"] = bm25_rrf_score + semantic_rrf_score

        sorted_scores = sorted(
            rrf_scores_map.items(),
            key=lambda item: item[1]["rrf_score"],
            reverse=True,
        )

        top = sorted_scores[:limit]

        results: list[dict] = []
        for doc_id, result in top:
            doc = result["doc"]
            search_result: dict = {}
            search_result["id"] = doc_id
            search_result["title"] = doc["title"]
            search_result["description"] = doc["description"]
            search_result["bm25_rank"] = result["bm25_rank"]
            search_result["semantic_rank"] = result["semantic_rank"]
            search_result["rrf_score"] = round(result["rrf_score"], SCORE_PRECISION)
            results.append(search_result)

        return results


def rrf_search_command(query, k_value, limit, enhancement: Optional[str] = None):
    documents = load_movies()
    hy_search = HybridSearch(documents)

    original_query = query
    updated_query = None
    if enhancement:
        updated_query = enhance_query(query, method=enhancement)
        if query != updated_query:
            print(f"Enhanced query ({enhancement}): '{query}' -> '{updated_query}'\n")
            query = updated_query

    results = hy_search.rrf_search(query, k_value, limit)

    print(f"Query: {query}")
    print(f"k: {k_value}")
    print("Results:")

    for i, result in enumerate(results, 1):
        print(f"\n{i}. {result['title']}")
        print(f"RRF Score: {result['rrf_score']:.4f}")
        print(
            f"BM25 Rank: {result['bm25_rank']}, Semantic Rank: {result['semantic_rank']}"
        )
        print(f"   {result['description'][:DOCUMENT_PREVIEW_LENGTH]}...")


def weighted_search_command(query, alpha, limit):
    documents = load_movies()
    hy_search = HybridSearch(documents)

    results = hy_search.weighted_search(query, alpha, limit)

    print(f"Query: {query}")
    print(f"Alpha: {alpha}")
    print("Results:")
    for i, result in enumerate(results, 1):
        print(f"\n{i}. {result['title']}")
        print(f"Hybrid Score: {result['hybrid_score']:.4f}")
        print(
            f"BM25: {result['bm25_score']:.4f}, Semantic: {result['semantic_score']:.4f}"
        )
        print(f"   {result['description']}...")
