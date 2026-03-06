import argparse
import json

from lib.hybrid_search import rrf_search_command
from lib.search_utils import load_golden_dataset


def main():
    parser = argparse.ArgumentParser(description="Search Evaluation CLI")
    parser.add_argument(
        "--limit",
        type=int,
        default=5,
        help="Number of results to evaluate (k for precision@k, recall@k)",
    )

    args = parser.parse_args()
    limit = args.limit

    # run evaluation logic here
    golden_data = load_golden_dataset()

    for test in golden_data:
        query = test["query"]
        rrf_results = rrf_search_command(query, 60, limit)
        relevant_docs = test["relevant_docs"]
        total_matched = 0
        retrieved = []
        relevant = []
        for item in rrf_results:
            if item["title"] in relevant_docs:
                total_matched += 1
                relevant.append(item["title"])
            retrieved.append(item["title"])
        precision_result = 0
        recall_result = 0
        if total_matched != 0:
            precision_result = total_matched / len(rrf_results)
            recall_result = len(relevant) / len(relevant_docs)
        print(f"- Query: {query}")
        print(f"  - Precision@{limit}: {precision_result:.4f}")
        print(f"  - Recall@{limit}: {recall_result:.4f}")
        print(f"  - Retrieved: {retrieved}")
        print(f"  - Relevant: {relevant}")


if __name__ == "__main__":
    main()
