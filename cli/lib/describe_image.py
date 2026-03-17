import mimetypes
import os

from dotenv import load_dotenv
from google import genai

load_dotenv()
api_key = os.getenv("gemini_api_key")
client = genai.Client(api_key=api_key)
model = "gemini-2.5-flash"


def describe_image_command(image_file: str, query):
    if not os.path.exists(image_file):
        raise FileNotFoundError(f"Selected image does not exists: {image_file}")

    mime, _ = mimetypes.guess_type(image_file)
    mime = mime or "image/jpeg"

    with open(image_file, "rb") as f:
        img = f.read()

    prompt = """Given the included image and text query, rewrite the text query to improve search results from a movie database. Make sure to:
        - Synthesize visual and textual information
        - Focus on movie-specific details (actors, scenes, style, etc.)
        - Return only the rewritten query, without any additional commentary"""

    parts = [
        prompt,
        genai.types.Part.from_bytes(data=img, mime_type=mime),
        query.strip(),
    ]
    response = client.models.generate_content(model=model, contents=parts)
    if response.text is None:
        raise RuntimeError("No text in AI Response")

    print(f"Rewritten query: {response.text.strip()}")
    if response.usage_metadata is not None:
        print(f"Total tokens:    {response.usage_metadata.total_token_count}")
