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


def citations_command(query, limit):
    search_results = rrf_search_command(query, RRF_K, None, None, limit)

    docs = str(
        [
            f"{result['title']} - {result['document'][:100]}"
            for result in search_results["results"]
        ]
    )

    prompt = f"""Answer the question or provide information based on the provided documents.

    This should be tailored to Hoopla users. Hoopla is a movie streaming service.

    If not enough information is available to give a good answer, say so but give as good of an answer as you can while citing the sources you have.

    Query: {query}

    Documents:
    {docs}

    Instructions:
    - Provide a comprehensive answer that addresses the query
    - Cite sources using [1], [2], etc. format when referencing information
    - If sources disagree, mention the different viewpoints
    - If the answer isn't in the documents, say "I don't have enough information"
    - Be direct and informative

    Answer:"""
    response = client.models.generate_content(model=model, contents=prompt)
    citation_response = (response.text or "").strip()
    print("Search Results:")
    for document in search_results["results"]:
        print(f" - {document['title']}")
    print(f"LLM Answer:\n {citation_response}")


def question_command(question, limit):
    search_results = rrf_search_command(question, RRF_K, None, None, limit)

    docs = str(
        [
            f"{result['title']} - {result['document']}"
            for result in search_results["results"]
        ]
    )

    prompt = f"""Answer the user's question based on the provided movies that are available on Hoopla.

    This should be tailored to Hoopla users. Hoopla is a movie streaming service.

    Question: {question}

    Documents:
    {docs}

    Instructions:
    - Answer questions directly and concisely
    - Be casual and conversational
    - Don't be cringe or hype-y
    - Talk like a normal person would in a chat conversation

    Answer:"""
    response = client.models.generate_content(model=model, contents=prompt)
    question_response = (response.text or "").strip()
    print("Search Results:")
    for document in search_results["results"]:
        print(f" - {document['title']}")
    print(f"Answer:\n {question_response}")


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
