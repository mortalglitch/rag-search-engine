#!/usr/bin/env python3

import argparse

from lib.keyword_search import (
    build_command,
    idf_command,
    search_command,
    tf_command,
    tf_idf_command,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Keyword Search CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    subparsers.add_parser(
        "build", help="Build the inverted index for quicker searching"
    )

    tf_parser = subparsers.add_parser(
        "tf", help="Return the amount of times a term appeared in a movie id"
    )
    tf_parser.add_argument(
        "document_id", type=int, help="ID of the movie you are looking for"
    )
    tf_parser.add_argument(
        "term", type=str, help="term you are looking for the count of"
    )

    idf_parser = subparsers.add_parser(
        "idf",
        help="Return the inverse document frequency for the provided term. (Term Rarity)",
    )
    idf_parser.add_argument("term", type=str, help="Term you would like the IDF for")

    tf_idf_parser = subparsers.add_parser(
        "tfidf",
        help="Return the TF-IDF for the provided term. (Term Frequency * Term Rarity)",
    )
    tf_idf_parser.add_argument(
        "document_id", type=int, help="ID of the movie you are looking for"
    )
    tf_idf_parser.add_argument("term", type=str, help="Term you would like the IDF for")

    search_parser = subparsers.add_parser("search", help="Search movies using BM25")
    search_parser.add_argument("query", type=str, help="Search query")

    args = parser.parse_args()

    match args.command:
        case "build":
            print("Building inverted index...")
            build_command()
            print("Inverted index built successfully.")
        case "idf":
            idf = idf_command(args.term)
            print(f"Inverse document frequency of '{args.term}': {idf:.2f}")
        case "search":
            print("Searching for:", args.query)
            results = search_command(args.query)
            for i, res in enumerate(results, 1):
                print(f"{i}. ({res['id']}) {res['title']}")
        case "tf":
            tf = tf_command(args.document_id, args.term)
            print(
                f"Term frequency of '{args.term}' in document '{args.document_id}': {tf}"
            )
        case "tfidf":
            tf_idf = tf_idf_command(args.document_id, args.term)
            print(
                f"TF-IDF score of '{args.term}' in document '{args.document_id}': {tf_idf:.2f}"
            )
        case _:
            parser.print_help()


if __name__ == "__main__":
    main()
