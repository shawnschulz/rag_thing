from langchain_community.document_loaders import AsyncChromiumLoader
from langchain_community.document_transformers import BeautifulSoupTransformer
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from flare_ai_rag.settings import settings
import json
import uuid
from flare_ai_rag.ai import EmbeddingTaskType, GeminiEmbedding
from qdrant_client.http.models import Distance, PointStruct, VectorParams
import structlog
from flare_ai_rag.retriever import QdrantRetriever, RetrieverConfig, generate_collection
from qdrant_client import QdrantClient
import hashlib
import requests
from bs4 import BeautifulSoup
import urllib.parse
import argparse
from google.generativeai.generative_models import ChatSession, GenerativeModel
from google.generativeai.client import configure
import asyncio

SYSTEM_INSTRUCTION = """
You are an expert extraction algorithm. 
Only extract relevant information from the text. 
If you do not know the value of an attribute asked to extract, 
return null for the attribute's value.

When helping users:
- Extract relevant information from web pages 
- If you are unsure or don't know of any useful information from a webpage, return null for it's value. There is no need to output any explanation of why you returned null, so do not output any explanation when this occurs.
- Always include the full text output of any articles or descriptions in documentation if they exist.
- You seek to inform people, always output the source information in a complete, yet still succint and well presented format.

"""
model_name="gemini-1.5-flash"
llm = GenerativeModel(
model_name=model_name,
system_instruction=SYSTEM_INSTRUCTION,
)



use_llm_extractor = False

logger = structlog.get_logger(__name__)

def setup_argparse():
    parser = argparse.ArgumentParser(description='Crawl or scrape urls into the qdrant database')
    
    parser.add_argument('--scrape', 
                        type=str,  # Can change to int, float, etc.
                        nargs='+',  # '+' means one or more arguments
                        help='Scrape any number of urls.')
    
    parser.add_argument('--crawl',
                        type=str,  # Can change to int, float, etc.
                        help='Crawl from the specified url. Takes only one url')
    parser.add_argument('--class_grep',
                        type=str,  # Can change to int, float, etc.curl -LsSf https://astral.sh/uv/install.sh | sh
                        help='Use to specify a specific keyword in webpages to enter links under')
    parser.add_argument('--llm_extraction',
                        action='store_true',
                        help="Use a large language model to extract information from web pages before storing in database. NOTE: I personally feel this doesn't work as well as just scraping the whole thing")
    
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

logger.info("Qdrant Client loaded")
# Just hard code this for now
embedding_client = GeminiEmbedding(GEMINI_API_KEY)


# These variables should eventually be handled by config files or user input
# Also would be nice if we tracked number of pages extracted by user defined
# schema type (i.e. we extracted 20 github pages and 10 blog articles)
# can make a dashboard for this later
urls = ['https://flare.network/news/shaping-the-future-of-blockchain-and-ai-flare-x-google-cloud-hackathon',  'https://flare.network/news/why-verifiable-ai-matters']

from langchain.chat_models import init_chat_model

def extract(content: str, use_llm_extractor=use_llm_extractor):
    if use_llm_extractor:
        return llm.generate_content(content);
    else:
        return content

import pprint
from langchain_text_splitters import RecursiveCharacterTextSplitter

# The string "NewsPage" works well for crawling https://flare.network/news
def crawl_webpage(url, use_llm_extractor=use_llm_extractor, max_pages=30, class_grep="NewsPage"):
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
            
            # Find all links under a class with a specific name
            elements_with_news_class = soup.find_all(class_=lambda c: c and class_grep in c)

            if class_grep != 'none':
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
                                payloads.append(scrape_with_playwright(absolute_link, use_llm_extractor))
                                to_visit.append(absolute_link)
                                count += 1
                                if count >= max_pages:
                                    return payloads, visited
            # Find all links 
            if class_grep == 'none':
                for link in soup.find_all('a', href=True):
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
def scrape_with_playwright(url, use_llm_extractor=use_llm_extractor):
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
    extracted_content = extract(content=splits[0].page_content, use_llm_extractor=use_llm_extractor)
    message_output = extracted_content
    payload = {}
    if use_llm_extractor:
        text_output = message_output.to_dict()['candidates'][0]['content']['parts'][0]['text']
        payload['metadata'] = str(message_output.usage_metadata)
    else: 
        text_output = splits[0].page_content
        payload['metadata'] = {'source':'website', 'extraction':'beautifulSoup'}
    #try: 
    #    payload = json.loads(text_output)
    #except json.JSONDecodeError:
    #    print("Failed to decode to json, adding to payload as text")

    # Just put the text in the bag
    payload['text'] = text_output 
    logger.info(f"Output of scraping: {text_output}")
    payload['filename'] = url
    #pprint.pprint(payload)
    return payload 

def string_to_int_hash(input_string, hash_bits=64):
    hash_object = hashlib.md5(input_string.encode())
    hex_dig = hash_object.hexdigest()
    int_value = int(hex_dig, 16) % (2 ** hash_bits)
    return int_value


def scrape_list_of_urls(urls):
    logger.info("Attempting to scrape a list of urls.")
    pass


def load_payloads_into_points(payloads, embedding_client=embedding_client):
    logger.info("Attempting to load scraped payloads into vectors.")
    points = []
    logger.info(f"Payloads: {payloads}")
    logger.info(f"Payloads type: {type(payloads)}")
    for payload in payloads:
        logger.info(f"Current payload: {payload}")
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

input_config = load_json(settings.input_path / "input_parameters.json")
retriever_config = RetrieverConfig.load(input_config["retriever_config"])
qdrant_client = QdrantClient(host=retriever_config.host, port=retriever_config.port)

# Upserts the points into the qdrant database
def upsert_database(points, qdrant_client=qdrant_client) -> None:
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

# Grab the qdrant client and 
    logger.info("Loading Qdrant client...")

    args = setup_argparse()
    if args.llm_extraction:
        use_llm_extractor = True
    if args.scrape is not None:
        urls = args.scrape
        payloads = [scrape_with_playwright(url) for url in urls]
        print(payloads)
        points = load_payloads_into_points(payloads)
        print(points)
        upsert_database(points, qdrant_client)
    if args.crawl is not None:
        source_url = args.crawl
        if args.class_grep is not None:
            payloads, _ = crawl_webpage(source_url, use_llm_extractor, 10, args.class_grep)
            points = load_payloads_into_points(payloads, embedding_client)
            upsert_database(points, qdrant_client)
        else:
            payloads, _ = crawl_webpage(source_url)
            points = load_payloads_into_points(payloads, embedding_client)
            upsert_database(points, qdrant_client)

