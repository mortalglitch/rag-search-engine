import json
import os

from dotenv import load_dotenv
from google import genai
from lib.hybrid_search import rrf_search_command
from lib.search_utils import DEFAULT_SEARCH_LIMIT, RRF_K

load_dotenv()
api_key = os.getenv("gemini_api_key")
client = genai.Client(api_key=api_key)
model = "gemini-2.5-flash"


def rag_command(query):
    search_results = rrf_search_command(query, RRF_K, None, None, DEFAULT_SEARCH_LIMIT)

    docs = str(
        [
            f"{result['title']} - {result['document'][:100]}"
            for result in search_results["results"]
        ]
    )

    prompt = (
        prompt
    ) = f"""Answer the question or provide information based on the provided documents. This should be tailored to Hoopla users. Hoopla is a movie streaming service.

    Query: {query}

    Documents:
    {docs}

    Provide a comprehensive answer that addresses the query:"""

    response = client.models.generate_content(model=model, contents=prompt)
    rag_response = (response.text or "").strip()
    print("Search Results:")
    for document in search_results["results"]:
        print(f" - {document['title']}")
    print(f"RAG Response:\n {rag_response}")


def summary_command(query: str, limit: int):
    search_results = rrf_search_command(query, RRF_K, None, None, limit)

    docs = str(
        [
            f"{result['title']} - {result['document'][:100]}"
            for result in search_results["results"]
        ]
    )

    prompt = f"""
    Provide information useful to this query by synthesizing information from multiple search results in detail.
    The goal is to provide comprehensive information so that users know what their options are.
    Your response should be information-dense and concise, with several key pieces of information about the genre, plot, etc. of each movie.
    This should be tailored to Hoopla users. Hoopla is a movie streaming service.
    Query: {query}
    Search Results:
    {docs}
    Provide a comprehensive 3–4 sentence answer that combines information from multiple sources:
    """

    response = client.models.generate_content(model=model, contents=prompt)
    summary_response_clean = (response.text or "").strip()
    print("Search Results:")
    for document in search_results["results"]:
        print(f" - {document['title']}")
    print(f"LLM Summary:\n {summary_response_clean}")
