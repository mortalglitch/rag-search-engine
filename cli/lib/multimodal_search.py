import os

from lib.search_utils import load_movies
from PIL import Image
from sentence_transformers import SentenceTransformer


class MultimodalSearch:
    def __init__(self, model_name="clip-ViT-B-32"):
        self.model = SentenceTransformer(model_name)

    def embed_image(self, image: str):
        if not os.path.exists(image):
            raise FileNotFoundError(f"Provided image not found {image}")

        # pyright doesn't like the following but it should work according to their own documentation.
        img_emb_result = self.model.encode([Image.open(image)])
        embedding = img_emb_result[0]
        return embedding


def verify_image_embedding(image: str):
    instance = MultimodalSearch()
    embedding = instance.embed_image(image)

    print(f"Embedding shape: {embedding.shape[0]} dimensions")
