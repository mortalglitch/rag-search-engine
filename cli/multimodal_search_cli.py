import argparse

from lib.multimodal_search import verify_image_embedding


def main():
    parser = argparse.ArgumentParser(description="Multimodal Search CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    verify_image_embedding_parser = subparsers.add_parser(
        "verify_image_embedding", help="Check image embedding"
    )
    verify_image_embedding_parser.add_argument(
        "image", type=str, help="Image file location"
    )

    args = parser.parse_args()

    match args.command:
        case "verify_image_embedding":
            verify_image_embedding(args.image)
        case _:
            parser.exit(2, parser.format_help())


if __name__ == "__main__":
    main()
