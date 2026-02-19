import argparse

from lib.search_utils import normalize_scores


def main() -> None:
    parser = argparse.ArgumentParser(description="Hybrid Search CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    normalizeparser = subparsers.add_parser(
        "normalize", help="Normalize a list of scores"
    )
    normalizeparser.add_argument(
        "scores", nargs="+", type=float, help="scores to normalize"
    )

    args = parser.parse_args()

    match args.command:
        case "normalize":
            results = normalize_scores(args.scores)
            for score in results:
                print(f"* {score:.4f}")
        case _:
            parser.print_help()


if __name__ == "__main__":
    main()
