import argparse

from lib.describe_image import describe_image_command


def main():
    parser = argparse.ArgumentParser(description="Image Description CLI")

    parser.add_argument(
        "--image", type=str, help="Image to feed to LLM for description."
    )
    parser.add_argument("--query", type=str, help="Query to submit with image.")

    args = parser.parse_args()
    describe_image_command(args.image, args.query)


if __name__ == "__main__":
    main()
