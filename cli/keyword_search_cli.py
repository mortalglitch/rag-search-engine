#!/usr/bin/env python3

import argparse
import json


def main() -> None:
    parser = argparse.ArgumentParser(description="Keyword Search CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    search_parser = subparsers.add_parser("search", help="Search movies using BM25")
    search_parser.add_argument("query", type=str, help="Search query")

    args = parser.parse_args()

    match args.command:
        case "search":
            print("Searching for:", args.query)
            count = 0
            results = {}
            with open("./data/movies.json", "r") as f:
                data = json.load(f)
                for item in data["movies"]:
                    if count <= 4:
                        if args.query in item["title"]:
                            count += 1
                            results[str(count)] = item["title"]
                    else:
                        break
            for key, value in results.items():
                print(f"{key}. {value}")

            pass
        case _:
            parser.print_help()


if __name__ == "__main__":
    main()
