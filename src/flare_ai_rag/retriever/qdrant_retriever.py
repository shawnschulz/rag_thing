from typing import override
from qdrant_client import models


from qdrant_client import QdrantClient

from flare_ai_rag.ai import EmbeddingTaskType, GeminiEmbedding
from flare_ai_rag.retriever.base import BaseRetriever
from flare_ai_rag.retriever.config import RetrieverConfig

from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.chat_models import init_chat_model
from google.generativeai.generative_models import ChatSession, GenerativeModel
from google.generativeai.client import configure
import structlog

logger = structlog.get_logger(__name__)

from fastembed import TextEmbedding

class QdrantRetriever(BaseRetriever):
    def __init__(
        self,
        client: QdrantClient,
        retriever_config: RetrieverConfig,
        embedding_client: GeminiEmbedding,
    ) -> None:
        """Initialize the QdrantRetriever."""
        self.client = client
        self.retriever_config = retriever_config
        self.embedding_client = embedding_client

    @override
    def semantic_search(self, query: str, top_k: int = 5) -> list[dict]:
        """
        Perform semantic search by converting the query into a vector
        and searching in Qdrant.

        :param query: The input query.
        :param top_k: Number of top results to return.
        :return: A list of dictionaries, each representing a retrieved document.
        """
        # Convert the query into a vector embedding using Gemini
        query_vector = self.embedding_client.embed_content(
            embedding_model="models/text-embedding-004",
            contents=query,
            task_type=EmbeddingTaskType.RETRIEVAL_QUERY,
        )

        # Idea: Use the query expansion prefetch to both improve the search (expand query)
        # and to rank answers
        # output results look something like this:
        # [
## ScoredPoint(
### id=0, version=0, score=1.5, 
### payload={'text': 'Oversampling in Qdrant search process defines how many extra vectors should be pre-selected using quantized index and then re-scored using original vectors to improve search quality.'}, 
### vector=None, shard_key=None, order_value=None), 
## ScoredPoint(
### id=75, version=0, score=0.8333333333333333, 
### payload={'text': 'Qdrant optimizes memory and search speed for sparse vectors by utilizing an inverted index structure to store vectors for each non-zero dimension. This approach allows Qdrant to efficiently represent sparse vectors, which are characterized by a high proportion of zeroes. By only storing information about non-zero dimensions, Qdrant reduces the memory footprint required to store sparse vectors and also speeds up search operations by focusing only on relevant dimensions during indexing and querying processes. This optimization ensures that Qdrant can handle sparse vectors effectively while maintaining efficient memory usage and search performance.'}, 
### vector=None, shard_key=None, order_value=None), 
## ScoredPoint(
### id=63, version=0, score=0.6666666666666666, 
### payload={'text': 'To optimize Qdrant for minimizing latency in search requests, you can set up the system to use as many cores as possible for a single request. This can be achieved by setting the number of segments in the collection to be equal to the number of cores in the system. By doing this, each segment will be processed in parallel, leading to a faster final result. This approach allows for the efficient utilization of system resources and can significantly reduce the time taken from the moment a request is submitted to the moment a response is received. By optimizing for latency in this manner, you can enhance the overall speed and responsiveness of the search functionality in Qdrant.'}, 
### vector=None, shard_key=None, order_value=None)
# ]
        SYSTEM_INSTRUCTION = f"""
        You are an expert to generate new questions from user questions. 

        Perform query expansion. If there are multiple common ways of phrasing a user question 
        or common synonyms for key words in the question, make sure to return multiple versions 
        of the query with the different phrasings. 

        If there are acronyms or words you are not familiar with, do not try to rephrase them. 

        Return 3 versions of the question: {query} 

        The Answer MUST be a list, clean your answer, trim them and don't add anything like bullets.
        """
        model_name="gemini-1.5-flash"
        llm = GenerativeModel(
        model_name=model_name,
        system_instruction=SYSTEM_INSTRUCTION,
        )
        embedding_model = TextEmbedding("BAAI/bge-small-en-v1.5")

        message_output= llm.generate_content(query)
        text_output = message_output.to_dict()['candidates'][0]['content']['parts'][0]['text']
        logger.info(f"DEBUG: output of gneerate content is: {text_output}")
        exp_embeddings_list = []
        for generated_query in text_output.splitlines():
            exp_embeddings_list.append(embedding_model.embed(generated_query))
        logger.info(f"DEBUG: exp_embeddings_list is: {exp_embeddings_list}")

        prefetch_array = []
        for embedding in exp_embeddings_list:
            temp = models.Prefetch(
                        query=embedding,
                        limit=3,
                    )
            prefetch_array.append(temp)
        results = self.client.query_points(
            collection_name=self.retriever_config.collection_name,
            prefetch=prefetch_array,
            query=models.FusionQuery(fusion=models.Fusion.RRF),
            limit=3,
            with_payload=True,
        )

        #So we could use the scores to display a list of the top relevant documents on the side
        #results = self.client.query_points(
        #    collection_name=self.retriever_config.collection_name,
        #    prefetch=[
        #        models.Prefetch(
        #            query=exp_embeddings_list[0],
        #            limit=3,
        #        ),
        #        models.Prefetch(
        #            query=exp_embeddings_list[1],
        #            limit=3,
        #        ),
        #        models.Prefetch(
        #            query=exp_embeddings_list[2],
        #            limit=3,
        #        ),
        #    ],
        #    query=models.FusionQuery(fusion=models.Fusion.RRF),
        #    limit=3,
        #    with_payload=True,
        #)
        # Search Qdrant for similar vectors.
        #results = self.client.search(
        #    collection_name=self.retriever_config.collection_name,
        #    query_vector=query_vector,
        #    limit=top_k,
        #)

        # Idea: Output the document relevance rankings in the UI somehow
        # Process and return results. To do this I think a jank way is to
        # set up a separate route/url and  send the request there from here,
        # since I can't see any other way. So it calls the other route here
        # and that should update the UI to have the document ranking 
        output = []
        for hit in results:
            if hit.payload:
                text = hit.payload.get("text", "")
                metadata = {
                    field: value
                    for field, value in hit.payload.items()
                    if field != "text"
                }
            else:
                text = ""
                metadata = ""
            output.append({"text": text, "score": hit.score, "metadata": metadata})
        return output
