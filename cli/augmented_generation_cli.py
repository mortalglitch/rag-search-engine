import argparse

from lib.augmented_generation import (
    citations_command,
    question_command,
    rag_command,
    summary_command,
)


def main():
    parser = argparse.ArgumentParser(description="Retrieval Augmented Generation CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    rag_parser = subparsers.add_parser(
        "rag", help="Perform RAG (search + generate answer)"
    )
    rag_parser.add_argument("query", type=str, help="Search query for RAG")

    summary_parser = subparsers.add_parser(
        "summarize", help="Summarize the results using a LLM"
    )
    summary_parser.add_argument(
        "query", type=str, help="query to search to later be summarized"
    )
    summary_parser.add_argument(
        "--limit", type=int, default=5, help="Number of results to return (default=5)"
    )

    citations_parser = subparsers.add_parser(
        "citations", help="User a LLM to generate results with citations."
    )
    citations_parser.add_argument(
        "query", type=str, help="query to search to later be summarized"
    )
    citations_parser.add_argument(
        "--limit", type=int, default=5, help="Number of results to return (default=5)"
    )

    question_parser = subparsers.add_parser(
        "question", help="question for the LLM regarding a movie"
    )
    question_parser.add_argument(
        "question", type=str, help="question to search to later be summarized"
    )
    question_parser.add_argument(
        "--limit", type=int, default=5, help="Number of results to return (default=5)"
    )

    args = parser.parse_args()

    match args.command:
        case "citations":
            query = args.query
            limit = args.limit
            citations_command(query, limit)
        case "question":
            question = args.question
            limit = args.limit
            question_command(question, limit)
        case "rag":
            query = args.query
            rag_command(query)
        case "summarize":
            query = args.query
            limit = args.limit
            summary_command(query, limit)

        case _:
            parser.print_help()


if __name__ == "__main__":
    main()
