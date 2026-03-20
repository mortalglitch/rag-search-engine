import os

from lib.search_utils import DOCUMENT_PREVIEW_LENGTH, load_movies
from lib.semantic_search import cosine_similarity
from numpy._typing import _32Bit
from PIL import Image
from sentence_transformers import SentenceTransformer


class MultimodalSearch:
    def __init__(self, documents, model_name="clip-ViT-B-32"):
        self.model = SentenceTransformer(model_name)
        self.documents = documents
        self.texts = setup_text(self.documents)
        self.text_embeddings = self.model.encode(self.texts, show_progress_bar=True)

    def embed_image(self, image: str):
        if not os.path.exists(image):
            raise FileNotFoundError(f"Provided image not found {image}")

        # pyright doesn't like the following but it should work according to their own documentation.
        img_emb_result = self.model.encode([Image.open(image)])
        embedding = img_emb_result[0]
        return embedding

    def search_with_image(self, image_path: str):
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Provided image not found {image_path}")

        image_embedding = self.embed_image(image_path)

        score_results = []
        for idx, text_embedding in enumerate(self.text_embeddings):
            similarity = cosine_similarity(image_embedding, text_embedding)
            score_result = self.documents[idx]
            score_result["score"] = similarity
            score_results.append(score_result)

        sorted_docs = sorted(
            score_results, key=lambda item: item["score"], reverse=True
        )

        top = sorted_docs[:5]
        return top


def image_search_command(image: str):
    documents = load_movies()
    instance = MultimodalSearch(documents)
    image_search_results = instance.search_with_image(image)
    for idx, result in enumerate(image_search_results, 1):
        print(f"Debug Value of Score: {result['score']}")
        print(f"{idx}. {result['title']} (similarity: {result['score']:.3f})")
        print(f"{result['description'][:DOCUMENT_PREVIEW_LENGTH]}...")


def setup_text(documents):
    results: list[str] = []
    for doc in documents:
        results.append(f"{doc['title']}: {doc['description']}")
    return results


# updated to support document load.
def verify_image_embedding(image: str):
    documents = load_movies()
    instance = MultimodalSearch(documents)
    embedding = instance.embed_image(image)

    print(f"Embedding shape: {embedding.shape[0]} dimensions")
