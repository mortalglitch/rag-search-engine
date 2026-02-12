#!/usr/bin/env python3

import argparse

from lib.search_utils import (
    DEFAULT_CHUNK_OVERLAP,
    DEFAULT_CHUNK_SIZE,
    DEFAULT_SEARCH_LIMIT,
    MAX_CHUNK_SIZE,
)
from lib.semantic_search import (
    chunk_text,
    embed_chunks,
    embed_query_text,
    embed_text,
    semantic_chunk_text,
    semantic_search,
    verify_embeddings,
    verify_model,
)


def main():
    parser = argparse.ArgumentParser(description="Semantic Search CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    subparsers.add_parser(
        "verify", help="Get information about the current loaded model"
    )

    subparsers.add_parser(
        "verify_embeddings", help="Ensure embeddings are properly loaded."
    )

    embed_chunks_parser = subparsers.add_parser(
        "embed_chunks", help="Embeds data as chunks"
    )

    embed_query_text_parser = subparsers.add_parser(
        "embedquery", help="Embeds a query and returns the results"
    )
    embed_query_text_parser.add_argument("query", type=str, help="query to embed")

    embed_text_parser = subparsers.add_parser(
        "embed_text", help="Get embedded text from input"
    )
    embed_text_parser.add_argument("text", type=str, help="Text to embed")

    search_parser = subparsers.add_parser(
        "search", help="Search movies using semantic search"
    )
    search_parser.add_argument("query", type=str, help="Search query")
    search_parser.add_argument(
        "--limit",
        type=int,
        nargs="?",
        default=DEFAULT_SEARCH_LIMIT,
        help="Number of search results to return",
    )

    semantic_chunk_parser = subparsers.add_parser(
        "semantic_chunk", help="Parse chunk semantically"
    )
    semantic_chunk_parser.add_argument("text", type=str, help="Text to chunk")
    semantic_chunk_parser.add_argument(
        "--max-chunk-size",
        type=int,
        nargs="?",
        default=MAX_CHUNK_SIZE,
        help="Max size allowed for chunks",
    )
    semantic_chunk_parser.add_argument(
        "--overlap",
        type=int,
        nargs="?",
        default=DEFAULT_CHUNK_OVERLAP,
        help="Size of desired chunk",
    )

    chunk_parser = subparsers.add_parser(
        "chunk", help="Split long descriptions for embedding"
    )
    chunk_parser.add_argument("text", type=str, help="Text to chunk")
    chunk_parser.add_argument(
        "--chunk-size",
        type=int,
        nargs="?",
        default=DEFAULT_CHUNK_SIZE,
        help="Size of desired chunk",
    )
    chunk_parser.add_argument(
        "--overlap",
        type=int,
        nargs="?",
        default=DEFAULT_CHUNK_OVERLAP,
        help="Size of desired chunk",
    )

    args = parser.parse_args()

    match args.command:
        case "chunk":
            results = chunk_text(args.text, args.chunk_size, args.overlap)
            # total_chars = sum(len(word) for sublist in results for word in sublist)
            # The above calculates the count of all letters in the list but doesn't count whitespace.
            total_chars = len(args.text)
            print(f"Chunking {total_chars} characters")
            for i, res in enumerate(results, 1):
                print(f"{i}. {' '.join(res)}")
        case "embed_chunks":
            embed_chunks()
        case "embedquery":
            embed_query_text(args.query)
        case "embed_text":
            embed_text(args.text)
        case "search":
            semantic_search(args.query, args.limit)
        case "semantic_chunk":
            results = semantic_chunk_text(args.text, args.max_chunk_size, args.overlap)
            total_chars = len(args.text)
            print(f"Semantically chunking {total_chars} characters")
            for i, res in enumerate(results, 1):
                print(f"{i}. {res}")
        case "verify":
            verify_model()
        case "verify_embeddings":
            verify_embeddings()
        case _:
            parser.print_help()


if __name__ == "__main__":
    main()
