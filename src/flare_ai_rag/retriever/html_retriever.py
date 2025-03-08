from langchain_community.document_loaders import AsyncChromiumLoader
from langchain_community.document_transformers import BeautifulSoupTransformer
from langchain_community.document_loaders import AsyncHtmlLoader
from langchain.chains import create_extraction_chain
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from typing import List, Optional




# Load the gemini API key from .env
import os
from dotenv import load_dotenv
load_dotenv()
GEMINI_API_KEY = os.getenv('GEMINI_API_KEY')

from langchain_google_genai import ChatGoogleGenerativeAI
#llm = ChatGoogleGenerativeAI(
#    model="gemini-1.5-flash",
#    temperature=0,
#    max_tokens=None,
#    timeout=None,
#    max_retries=2,
#)



# These variables should eventually be handled by config files or user input
# Also would be nice if we tracked number of pages extracted by user defined
# schema type (i.e. we extracted 20 github pages and 10 blog articles)
# can make a dashboard for this later
schema = {
    "properties": {
        "article_title": {"type": "string"},
        "article_summary": {"type": "string"},
    },
    "required": ["article_title", "article_summary"],
}
urls = ['https://dev.flare.network/network/getting-started']
data_file_path = "/home/bankerz/Programs/flare-hackathon/rag_thing/src/data"

prompt_template = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are an expert extraction algorithm. "
            "Only extract relevant information from the text. "
            "If you do not know the value of an attribute asked to extract, "
            "return null for the attribute's value.",
        ),
        # Please see the how-to about improving performance with
        # reference examples.
        # MessagesPlaceholder('examples'),
        ("user", "{text}"),
    ]
)

from langchain.chat_models import init_chat_model
llm = init_chat_model("gemini-1.5-flash", model_provider="google_genai")
#llm = init_chat_model("gemini-2.0-flash-001", model_provider="google_vertexai")
structured_llm = llm.with_structured_output(schema=schema)

def extract(content: str, schema: dict):
    return prompt_template.invoke({"text": content}) 

import pprint
from langchain_text_splitters import RecursiveCharacterTextSplitter

def scrape_with_playwright(urls, schema):
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
    extracted_content = extract(schema=schema, content=splits[0].page_content)
    pprint.pprint(extracted_content)
    return extracted_content

extracted_content = scrape_with_playwright(urls, schema=schema)

## Load HTML
#loader = AsyncHtmlLoader(urls)
#html = loader.load()
##print(html)
## Transform
#bs_transformer = BeautifulSoupTransformer()
#docs_transformed = bs_transformer.transform_documents(html, tags_to_extract=["article"])
#print(docs_transformed[0].page_content[0:])
#

