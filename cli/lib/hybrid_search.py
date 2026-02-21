import os

from lib.search_utils import (
    DEFAULT_HYBRID_ALPHA,
    DEFAULT_HYBRID_LIMIT,
    DOCUMENT_PREVIEW_LENGTH,
    SCORE_PRECISION,
    hybrid_score,
    load_movies,
    normalize_scores,
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

    def rrf_search(self, query, k, limit=10):
        raise NotImplementedError("RRF hybrid search is not implemented yet.")

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
