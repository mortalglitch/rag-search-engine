from sentence_transformers import SentenceTransformer


class SemanticSearch:
    def __init__(self) -> None:
        self.model = SentenceTransformer("all-MiniLM-L6-V2")

    def generate_embedding(self, text):
        if not text.strip():
            raise ValueError("Text must not contain whitespace")
        elif len(text) == 0:
            raise ValueError("Text must not be empty")

        text_store = []
        text_store.append(text)

        result = self.model.encode(text_store)

        return result[0]


def embed_text(text):
    sem_search = SemanticSearch()
    embedded_result = sem_search.generate_embedding(text)

    print(f"Text: {text}")
    print(f"First 3 dimensions: {embedded_result[:3]}")
    print(f"Dimensions: {embedded_result.shape[0]}")


def verify_model():
    new_semantic = SemanticSearch()
    print(f"Model loaded:{new_semantic.model}")
    print(f"Max sequence length: {new_semantic.model.max_seq_length}")
    return
