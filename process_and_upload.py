import os
import requests
from typing import List, Dict
from dotenv import load_dotenv
from anthropic import Anthropic
import voyageai
from PyPDF2 import PdfReader
from bs4 import BeautifulSoup
from tqdm import tqdm
import time
from astrapy.db import AstraDB

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
        self.collection_name = collection_name
        self.astra_db = AstraDB(token=ASTRA_DB_APPLICATION_TOKEN, api_endpoint=ASTRA_DB_API_ENDPOINT)

        try:
            self.astra_db.delete_collection(collection_name)
            print(f"Deleted existing collection: {collection_name}")
        except Exception as e:
            print(f"No existing collection to delete or error occurred: {e}")

        # Create a new collection
        self.collection = self.astra_db.create_collection(collection_name, dimension=1536)
        print(f"Created new collection: {collection_name}")

    def load_data(self, data):
        texts = [f"{item['chunk_heading']}\n\n{item['text']}\n\n{item['summary']}" for item in data]
        batch_size = 128
        for i in tqdm(range(0, len(texts), batch_size), desc="Generating embeddings and storing"):
            batch = texts[i:i+batch_size]
            embeddings = voyage_client.embed(batch, model="voyage-large-2").embeddings
            documents = [
                {
                    "$vector": embedding,
                    **data[i+j]
                } for j, embedding in enumerate(embeddings)
            ]
            self.collection.insert_many(documents)

def load_pdf_content(file_path: str) -> str:
    with open(file_path, 'rb') as file:
        reader = PdfReader(file)
        text = ""
        for page in tqdm(reader.pages, desc=f"Reading {os.path.basename(file_path)}", leave=False):
            text += page.extract_text() + "\n"
    return text

def load_url_content(url: str) -> str:
    response = requests.get(url)
    soup = BeautifulSoup(response.content, 'html.parser')
    return soup.get_text()

def chunk_text(text: str, chunk_size: int = 500, overlap: int = 250) -> List[str]:
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end]
        chunks.append(chunk)
        start += chunk_size - overlap
    return chunks

def generate_summary(text: str) -> str:
    prompt = f"""
    Please provide a brief summary of the following content in 2-3 sentences. The summary should capture the key points and be concise.

    Content to summarize:
    {text}

    Summary:
    """
    response = anthropic_client.messages.create(
        model="claude-3-sonnet-20240229",
        max_tokens=150,
        messages=[
            {"role": "user", "content": prompt}
        ],
        temperature=0
    )
    return response.content[0].text.strip()

def process_documents(pdf_directory: str, urls: List[str]) -> List[Dict]:
    processed_data = []

    # Process PDFs
    pdf_files = [f for f in os.listdir(pdf_directory) if f.lower().endswith('.pdf')]
    for filename in tqdm(pdf_files, desc="Processing PDFs"):
        file_path = os.path.join(pdf_directory, filename)
        content = load_pdf_content(file_path)
        chunks = chunk_text(content)
        for i, chunk in enumerate(tqdm(chunks, desc=f"Summarizing chunks of {filename}", leave=False)):
            summary = generate_summary(chunk)
            processed_data.append({
                "chunk_heading": f"PDF: {filename} - Chunk {i + 1}",
                "text": chunk,
                "summary": summary,
                "source": file_path
            })

    # Process URLs
    for url in tqdm(urls, desc="Processing URLs"):
        content = load_url_content(url)
        chunks = chunk_text(content)
        for i, chunk in enumerate(tqdm(chunks, desc=f"Summarizing chunks of {url}", leave=False)):
            summary = generate_summary(chunk)
            processed_data.append({
                "chunk_heading": f"URL: {url} - Chunk {i + 1}",
                "text": chunk,
                "summary": summary,
                "source": url
            })

    return processed_data

def main():
    pdf_directory = 'data_pdf'
    urls = [
        # Add URLs as needed
    ]

    print("Starting document processing...")
    start_time = time.time()
    processed_data = process_documents(pdf_directory, urls)
    processing_time = time.time() - start_time
    print(f"Document processing completed in {processing_time:.2f} seconds.")

    print("Initializing and loading vector database...")
    start_time = time.time()
    db = AstraDBVectorStore("vyce_docs")
    db.load_data(processed_data)
    loading_time = time.time() - start_time
    print(f"Vector database loaded in {loading_time:.2f} seconds.")

if __name__ == "__main__":
    main()
