from langchain_community.document_loaders import AsyncChromiumLoader
from langchain_community.document_transformers import BeautifulSoupTransformer
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from Document import Document, Data
from settings import settings
import json
import uuid
from ai import EmbeddingTaskType, GeminiEmbedding
from qdrant_client.http.models import Distance, PointStruct, VectorParams
import structlog
from flare_ai_rag.retriever import QdrantRetriever, RetrieverConfig, generate_collection
from qdrant_client import QdrantClient
import hashlib
import requests
from bs4 import BeautifulSoup
import urllib.parse
import argparse

def setup_argparse():
    parser = argparse.ArgumentParser(description='Crawl or scrape urls into the qdrant database')
    
    # Add an argument that takes a single value
    parser.add_argument('--scrape', 
                        type=str,  # Can change to int, float, etc.
                        nargs='+',  # '+' means one or more arguments
                        help='Scrape any number of urls.')
    
    # Add an argument that can take multiple values (a list)
    parser.add_argument('--crawl',
                        type=str,  # Can change to int, float, etc.
                        help='Crawl from the specified url. Takes only one url')
    
    return parser.parse_args()


# Load the gemini API key from .env
import os
from dotenv import load_dotenv
load_dotenv()
GEMINI_API_KEY = os.getenv('GEMINI_API_KEY')

from langchain_google_genai import ChatGoogleGenerativeAI

def load_json(file_path) -> dict:
    """Read the selected model IDs from a JSON file."""
    with file_path.open() as f:
        return json.load(f)


# These variables should eventually be handled by config files or user input
# Also would be nice if we tracked number of pages extracted by user defined
# schema type (i.e. we extracted 20 github pages and 10 blog articles)
# can make a dashboard for this later
urls = ['https://flare.network/news/shaping-the-future-of-blockchain-and-ai-flare-x-google-cloud-hackathon',  'https://flare.network/news/why-verifiable-ai-matters']

from langchain.chat_models import init_chat_model

def extract(content: str):
    return prompt_template.invoke({"text": content}) 

import pprint
from langchain_text_splitters import RecursiveCharacterTextSplitter

def crawl_webpage(url, max_pages=10):
    """
    Thanks Claude!

    Crawl a webpage and its links using BeautifulSoup and requests
    
    Args:
        url: Starting URL to crawl
        max_pages: Maximum number of pages to crawl (to prevent infinite crawling)
    
    Returns:
        Dictionary with URLs as keys and their content/title as values
    """
    logger.info(f"Begnning to crawl webpage")
    # Keep track of visited pages to avoid loops
    visited = {}
    # Queue of URLs to visit
    to_visit = [url]
    # Counter for number of pages visited
    count = 0
    payloads = []
    
    while to_visit and count < max_pages:
        # Get the next URL to visit
        current_url = to_visit.pop(0)
        
        # Skip if already visited
        if current_url in visited:
            continue
        
        logger.info(f"Crawling: {current_url}")
        
        try:
            # Make the HTTP request
            response = requests.get(current_url, headers={
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
            })
            
            # Check if request was successful
            response.raise_for_status()
            
            # Parse the HTML content
            soup = BeautifulSoup(response.text, 'html.parser')
            
            
            # Store the page info
            visited[current_url] = {
                'links': []
            }
            
            # Find all links on the page
            elements_with_news_class = soup.find_all(class_=lambda c: c and "NewsPage" in c)

            for element in elements_with_news_class:
                for link in element.find_all('a', href=True):
                    href = link['href']
                    
                    # Skip empty links, javascript, and anchors
                    if not href or href.startswith('javascript:') or href.startswith('#'):
                        continue
                    
                    # Convert relative URLs to absolute
                    absolute_link = urllib.parse.urljoin(current_url, href)
                    
                    # Only follow links to the same domain
                    if urllib.parse.urlparse(absolute_link).netloc == urllib.parse.urlparse(current_url).netloc:
                        # Add link to the list of links on the current page
                        visited[current_url]['links'].append(absolute_link)
                        
                        # Add to the queue if not visited yet
                        if absolute_link not in visited and absolute_link not in to_visit:
                            # Scrape the webpage if it wasn't visited yet
                            payloads.append(scrape_with_playwright(absolute_link))
                            to_visit.append(absolute_link)
                            count += 1
                            if count >= max_pages:
                                return payloads, visited
            
        except Exception as e:
            print(f"Error crawling {current_url}: {e}")
            # Mark as visited to avoid retrying
            visited[current_url] = {
                'title': f"Error: {str(e)}",
                'links': []
            }
    
    return payloads, visited
# Takes a url and returns a payload to be upserted into qdrant vector database
# Uses gemini for content extraction (albeit it doesn't particularly listen to me)
def scrape_with_playwright(url):
    logger.info("Attempting to scrape urls.")
    urls = [url]
    loader = AsyncChromiumLoader(urls)
    docs = loader.load()
    bs_transformer = BeautifulSoupTransformer()
    docs_transformed = bs_transformer.transform_documents(
        docs, tags_to_extract=["span", "article"]
    )
    print("Extracting content with LLM")

    # Grab the first 1000 tokens of the site
    splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        chunk_size=1000, chunk_overlap=0
    )
    splits = splitter.split_documents(docs_transformed)

    # Process the first split
    extracted_content = extract(content=splits[0].page_content)
    message_output = extracted_content.to_messages()[1] 
    pprint.pprint(message_output.content)
    payload = {
        "filename": url,
        "metadata": message_output.response_metadata,
        "text": message_output.content,
    }
    return payload 

def string_to_int_hash(input_string, hash_bits=64):
    hash_object = hashlib.md5(input_string.encode())
    hex_dig = hash_object.hexdigest()
    int_value = int(hex_dig, 16) % (2 ** hash_bits)
    return int_value


def scrape_list_of_urls(urls):
    logger.info("Attempting to scrape a list of urls.")
    pass


def load_payloads_into_points(embedding_client, payloads):
    logger.info("Attempting to load scraped payloads into vectors.")
    points = []
    for payload in payloads:
        json_string = json.dumps(payload)

        generated_uuid = uuid.uuid5(uuid.NAMESPACE_URL, json_string)
        hashed_id = string_to_int_hash(json_string)
        content = payload['text']
        embedding = embedding_client.embed_content(
            embedding_model=retriever_config.embedding_model,
            task_type=EmbeddingTaskType.RETRIEVAL_DOCUMENT,
            contents=content,
            title=str(payload['filename']),
        )

        point = PointStruct(
            id=hashed_id,  # Using integer ID starting from 1
            vector=embedding,
            payload=payload,
        )
        points.append(point)
    return points


# Upserts the points into the qdrant database
def upsert_database(qdrant_client, points) -> None:
    logger.info("Attempting to load vectors into database.")
    qdrant_client.upsert(
        collection_name=retriever_config.collection_name,
        points=points,
    )
    logger.info(
        "Documents inserted into Qdrant successfully.",
        collection_name=retriever_config.collection_name,
        num_points=len(points),
    )
    pass

if __name__ == "__main__":
    
    prompt_template = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You are an expert extraction algorithm. "
                "Only extract relevant information from the text. "
                "If you do not know the value of an attribute asked to extract, "
                "return null for the attribute's value.",
            ),
            ("user", "{text}"),
        ]
    )
    llm = init_chat_model("gemini-1.5-flash", model_provider="google_genai")
    structured_llm = llm.with_structured_output(schema=Document)
    logger = structlog.get_logger(__name__)
# Grab the qdrant client and 
    input_config = load_json(settings.input_path / "input_parameters.json")
    logger.info("Loading Qdrant client...")
    retriever_config = RetrieverConfig.load(input_config["retriever_config"])
    qdrant_client = QdrantClient(host=retriever_config.host, port=retriever_config.port)
    logger.info("Qdrant Client loaded")
# Just hard code this for now
    embedding_client = GeminiEmbedding(GEMINI_API_KEY)

    args = setup_argparse()
    if args.scrape is not None:
        urls = args.scrape
        payloads = [scrape_with_playwright(url) for url in urls]
        print(payloads)
        points = load_payloads_into_points(embedding_client, payloads)
        print(points)
        upsert_database(qdrant_client, points)
    if args.crawl is not None:
        source_url = args.crawl
        payloads, _ = crawl_webpage(source_url)
        points = load_payloads_into_points(embedding_client, payloads)
        upsert_database(qdrant_client, points)

