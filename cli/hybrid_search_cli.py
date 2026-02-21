import argparse

from lib.hybrid_search import rrf_search_command, weighted_search_command
from lib.search_utils import (
    DEFAULT_HYBRID_ALPHA,
    DEFAULT_HYBRID_LIMIT,
    DEFAULT_K,
    normalize_scores,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Hybrid Search CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    normalizeparser = subparsers.add_parser(
        "normalize", help="Normalize a list of scores"
    )
    normalizeparser.add_argument(
        "scores", nargs="+", type=float, help="scores to normalize"
    )

    weighted_search = subparsers.add_parser(
        "weighted-search", help="Normalize a list of scores"
    )
    weighted_search.add_argument("query", type=str, help="Search text")
    weighted_search.add_argument(
        "--alpha",
        type=float,
        nargs="?",
        default=DEFAULT_HYBRID_ALPHA,
        help="Search alpha",
    )
    weighted_search.add_argument(
        "--limit",
        type=int,
        nargs="?",
        default=DEFAULT_HYBRID_LIMIT,
        help="Number of search results to return",
    )

    rrf_search_parser = subparsers.add_parser(
        "rrf-search", help="Get results from a RRF method"
    )
    rrf_search_parser.add_argument("query", type=str, help="Search text")
    rrf_search_parser.add_argument(
        "-k",
        type=int,
        nargs="?",
        default=DEFAULT_K,
        help="Search K value(weight)",
    )
    rrf_search_parser.add_argument(
        "--limit",
        type=int,
        nargs="?",
        default=DEFAULT_HYBRID_LIMIT,
        help="Number of search results to return",
    )

    args = parser.parse_args()

    match args.command:
        case "normalize":
            results = normalize_scores(args.scores)
            for score in results:
                print(f"* {score:.4f}")
        case "rrf-search":
            rrf_search_command(args.query, args.k, args.limit)
        case "weighted-search":
            weighted_search_command(args.query, args.alpha, args.limit)
        case _:
            parser.print_help()


if __name__ == "__main__":
    main()
