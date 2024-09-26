from dotenv import load_dotenv
from langchain_community.embeddings.openai import OpenAIEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain_astradb import AstraDBVectorStore
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_core.documents import Document
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
import requests
import os

# Load environment variables from .env file
load_dotenv()

ASTRA_DB_APPLICATION_TOKEN = os.environ.get("ASTRA_DB_APPLICATION_TOKEN")
ASTRA_DB_API_ENDPOINT = os.environ.get("ASTRA_DB_API_ENDPOINT")
ASTRA_DB_KEYSPACE = os.environ.get("ASTRA_DB_KEYSPACE")
ASTRA_DB_COLLECTION = os.environ.get("ASTRA_DB_COLLECTION")
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")

# Configuration
pdf_directory = 'data_pdf'
urls = [
    "https://www.vyce.io/",
    # Add more URLs as needed
]


def load_pdf_files(directory):
    documents = []
    for filename in os.listdir(directory):
        if filename.lower().endswith('.pdf'):
            file_path = os.path.join(directory, filename)
            loader = PyPDFLoader(file_path)
            documents.extend(loader.load())
    return documents


def fetch_content_from_url(url):
    response = requests.get(url)
    response.raise_for_status()
    return Document(page_content=response.text, metadata={"source": url})


def load_url_content(urls):
    return [fetch_content_from_url(url) for url in urls]


def split_text(documents):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=250,
        length_function=len,
        add_start_index=True,
    )
    chunks = text_splitter.split_documents(documents)
    print(f'Split {len(documents)} documents into {len(chunks)} chunks.')
    return chunks


def save_to_astradb(chunks):
    embedding = OpenAIEmbeddings()
    vstore = AstraDBVectorStore(
        embedding=embedding,
        collection_name=ASTRA_DB_COLLECTION,
        token=ASTRA_DB_APPLICATION_TOKEN,
        api_endpoint=ASTRA_DB_API_ENDPOINT,
    )
    inserted_ids = vstore.add_documents(chunks)
    print(f"\nInserted {len(inserted_ids)} documents.")
    return vstore


def main(pdf_directory=pdf_directory, urls=urls):
    # Load and process PDFs
    pdf_documents = load_pdf_files(pdf_directory)

    # Load and process URLs
    url_documents = load_url_content(urls)

    # Combine all documents
    all_documents = pdf_documents + url_documents

    # Split text into chunks
    chunks = split_text(all_documents)

    # Save to AstraDB
    vstore = save_to_astradb(chunks)

    # Start the chatbot
    start_chatbot(vstore)


def start_chatbot(vstore):
    retriever = vstore.as_retriever(search_kwargs={'k': 3, 'similarity_threshold': 0.7})

    memory = ConversationBufferMemory(
        memory_key="chat_history",
        return_messages=True,
        output_key="answer"  # Specify which output to store in memory
    )

    prompt_template = """
    You are acting as an AI customer support assistant chatbot for Vyce and the construction industry, using the context and chat history you have access to. You are chatting with a human user who may be asking for help about Vyce's products, services, or general construction industry information. When responding to the user, aim to provide concise and helpful responses while maintaining a polite, professional, and slightly bubbly tone. Feel free to use emojis occasionally to enhance engagement.

    To help you answer the user's question, we have retrieved the following context information:
    {context}

    Please provide responses that primarily uses the information you have been given. If the information is not sufficient or relevant for answering the question, you can state that you don't have enough information to fully answer the query.

    Important: Always include relevant links that start with https: when available. If you have access to a URL that relates to the user's query or your response, make sure to include it in your answer.

    Context: {context}
    Chat History: {chat_history}
    Human: {question}
    AI Assistant:"""

    qa_chain = ConversationalRetrievalChain.from_llm(
        llm=ChatOpenAI(openai_api_key=OPENAI_API_KEY, model_name="gpt-4-turbo", temperature=0.1),
        retriever=retriever,
        memory=memory,
        verbose=True,
        return_source_documents=True,
        combine_docs_chain_kwargs={"prompt": ChatPromptTemplate.from_template(prompt_template)}
    )

    print("Hey! My name is Laila, and I'm your automated assistant. Feel free to ask me anything "
          "related to: Vyce, general construction and even more diverse topics.")
    while True:
        query = input("You: ")
        if query.lower() == 'exit':
            break
        result = qa_chain({"question": query})
        print(f"AI Assistant: {result['answer']}")


if __name__ == "__main__":
    main()