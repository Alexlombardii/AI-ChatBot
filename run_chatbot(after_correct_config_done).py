import os
from dotenv import load_dotenv
from anthropic import Anthropic
import voyageai
from astrapy.db import AstraDB
from typing import List, Dict, Tuple
import time
from config import IDENTITY, TASK_SPECIFIC_INSTRUCTIONS, MODEL

# Load environment variables
load_dotenv()

# Initialize clients
anthropic_client = Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
voyage_client = voyageai.Client(api_key=os.getenv("VOYAGE_API_KEY"))

# AstraDB configuration
ASTRA_DB_APPLICATION_TOKEN = os.getenv('ASTRA_DB_APPLICATION_TOKEN')
ASTRA_DB_API_ENDPOINT = os.getenv('ASTRA_DB_API_ENDPOINT')

class AstraDBVectorStore:
    def __init__(self, collection_name):
        self.astra_db = AstraDB(token=ASTRA_DB_APPLICATION_TOKEN, api_endpoint=ASTRA_DB_API_ENDPOINT)
        self.collection = self.astra_db.collection(collection_name)

    def search(self, query, k=3):
        query_embedding = voyage_client.embed([query], model="voyage-large-2").embeddings[0]
        results = self.collection.vector_find(
            vector=query_embedding,
            limit=k,
            fields=["chunk_heading", "text", "summary", "source"]
        )
        return [
            {
                "metadata": {
                    "chunk_heading": doc['chunk_heading'],
                    "text": doc['text'],
                    "summary": doc['summary'],
                    "source": doc['source']
                },
                "similarity": doc['$similarity']
            }
            for doc in results
        ]

def rerank_results(query: str, results: List[Dict], k: int = 5) -> List[Dict]:
    summaries = [f"[{i}] Document Summary: {result['metadata']['summary']}" for i, result in enumerate(results)]
    joined_summaries = "\n\n".join(summaries)

    prompt = f"""
    Query: {query}
    You are about to be given a group of documents, each preceded by its index number in square brackets. Your task is to select the {k} most relevant documents from the list to help us answer the query.

    <documents>
    {joined_summaries}
    </documents>

    Output only the indices of {k} most relevant documents in order of relevance, separated by commas, enclosed in XML tags here:
    <relevant_indices>put the numbers of your indices here, separated by commas</relevant_indices>
    """
    response = anthropic_client.messages.create(
        model=MODEL,
        max_tokens=50,
        messages=[{"role": "user", "content": prompt}, {"role": "assistant", "content": "<relevant_indices>"}],
        temperature=0,
        stop_sequences=["</relevant_indices>"]
    )

    indices_str = response.content[0].text.strip()
    relevant_indices = [int(idx.strip()) for idx in indices_str.split(',') if idx.strip().isdigit()]

    if not relevant_indices:
        relevant_indices = list(range(min(k, len(results))))

    relevant_indices = [idx for idx in relevant_indices if idx < len(results)]
    reranked_results = [results[idx] for idx in relevant_indices[:k]]

    for i, result in enumerate(reranked_results):
        result['relevance_score'] = 100 - i

    return reranked_results

def retrieve_advanced(query: str, db: AstraDBVectorStore, k: int = 2, initial_k: int = 10) -> Tuple[List[Dict], str]:
    initial_results = db.search(query, k=initial_k)
    reranked_results = rerank_results(query, initial_results, k=k)

    new_context = ""
    for result in reranked_results:
        chunk = result['metadata']
        new_context += f"\n <document> \n {chunk['chunk_heading']}\n\n{chunk['text']} \n </document> \n"

    return reranked_results, new_context

def answer_query(query: str, context: str) -> str:
    prompt = f"""
    {IDENTITY}

    {TASK_SPECIFIC_INSTRUCTIONS}

    Answer the following query based on the provided context. If the information is not in the context, say you don't have enough information to answer.

    Query: {query}

    Context:
    {context}

    Answer:
    """
    response = anthropic_client.messages.create(
        model=MODEL,
        max_tokens=500,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.1
    )
    return response.content[0].text.strip()

def main():
    db = AstraDBVectorStore("vyce_docs")

    print("Chatbot initialized. Type 'exit' to end the conversation.")

    # Initialize the conversation with TASK_SPECIFIC_INSTRUCTIONS
    context = TASK_SPECIFIC_INSTRUCTIONS
    print("AI: Hello! I'm Laila, an AI assistant for Vyce in the construction industry. How can I help you today?")

    while True:
        query = input("You: ")
        if query.lower() == 'exit':
            break

        start_time = time.time()
        results, new_context = retrieve_advanced(query, db)
        retrieval_time = time.time() - start_time

        # Combine the initial context with the new context
        full_context = context + "\n" + new_context

        start_time = time.time()
        answer = answer_query(query, full_context)
        answer_time = time.time() - start_time
        print(f"AI: {answer}")

        # Update the context for the next iteration
        context = full_context

if __name__ == "__main__":
    main()
