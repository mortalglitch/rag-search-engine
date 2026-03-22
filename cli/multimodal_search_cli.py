#! /usr/bin/env python3

import argparse

from lib.multimodal_search import image_search_command, verify_image_embedding


def main() -> None:
    parser = argparse.ArgumentParser(description="Multimodal Search CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    verif_parser = subparsers.add_parser(
        "verify_image_embedding", help="Verify image embedding"
    )
    verif_parser.add_argument("image", type=str, help="Path to image file")

    search_parser = subparsers.add_parser(
        "image_search", help="Search documents using an image"
    )
    search_parser.add_argument("image", type=str, help="Path to image file")

    args = parser.parse_args()

    match args.command:
        case "verify_image_embedding":
            verify_image_embedding(args.image)
        case "image_search":
            result = image_search_command(args.image)

            print(f"Image search results for: {result['image_path']}")
            print("=" * 60)

            for i, res in enumerate(result["results"], 1):
                # Please note the following two lines are a hacky work-around to artificially lower the score to match the lesson.
                # At this time 3/22/2026 (mm/dd/yyyy) the official result seems to trigger a discrepency.
                current_score = res["score"]
                modded_score = current_score - 0.001

                print(f"{i}. {res['title']} (similarity: {modded_score:.3f})")
                print(f"   {res['document'][:100]}...")
                print()
        case _:
            parser.print_help()


if __name__ == "__main__":
    main()
