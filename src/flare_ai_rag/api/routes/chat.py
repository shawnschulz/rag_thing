import structlog
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from flare_ai_rag.ai import GeminiProvider
from flare_ai_rag.attestation import Vtpm, VtpmAttestationError
from flare_ai_rag.prompts import PromptService, SemanticRouterResponse
from flare_ai_rag.prompts.schemas import ExtractionPipelineResponse
from flare_ai_rag.responder import GeminiResponder
from flare_ai_rag.retriever import QdrantRetriever
from flare_ai_rag.router import GeminiRouter
from flare_ai_rag.html_retriever import scrape_with_playwright
from flare_ai_rag.html_retriever import crawl_webpage 

import json

logger = structlog.get_logger(__name__)
router = APIRouter()


class ChatMessage(BaseModel):
    """
    Pydantic model for chat message validation.

    Attributes:
        message (str): The chat message content, must not be empty
    """

    message: str = Field(..., min_length=1)


class ChatRouter:
    """
    A simple chat router that processes incoming messages using the RAG pipeline.

    It wraps the existing query classification, document retrieval, and response
    generation components to handle a conversation in a single endpoint.
    """

    def __init__(  # noqa: PLR0913
        self,
        router: APIRouter,
        ai: GeminiProvider,
        query_router: GeminiRouter,
        retriever: QdrantRetriever,
        responder: GeminiResponder,
        attestation: Vtpm,
        prompts: PromptService,
    ) -> None:
        """
        Initialize the ChatRouter.

        Args:
            router (APIRouter): FastAPI router to attach endpoints.
            ai (GeminiProvider): AI client used by a simple semantic router
                to determine if an attestation was requested or if RAG
                pipeline should be used.
            query_router: RAG Component that classifies the query.
            retriever: RAG Component that retrieves relevant documents.
            responder: RAG Component that generates a response.
            attestation (Vtpm): Provider for attestation services
            prompts (PromptService): Service for managing prompts
        """
        self._router = router
        self.ai = ai
        self.query_router = query_router
        self.retriever = retriever
        self.responder = responder
        self.attestation = attestation
        self.prompts = prompts
        self.logger = logger.bind(router="chat")
        self._setup_routes()

    def _setup_routes(self) -> None:
        """
        Set up FastAPI routes for the chat endpoint.
        """

        @self._router.post("/")
        async def chat(message: ChatMessage) -> dict[str, str] | None:  # pyright: ignore [reportUnusedFunction]
            """
            Process a chat message through the RAG pipeline.
            Returns a response containing the query classification and the answer.
            """
            try:
                self.logger.debug("Received chat message", message=message.message)

                # If attestation has previously been requested:
                if self.attestation.attestation_requested:
                    try:
                        resp = self.attestation.get_token([message.message])
                    except VtpmAttestationError as e:
                        resp = f"The attestation failed with  error:\n{e.args[0]}"
                    self.attestation.attestation_requested = False
                    return {"response": resp}

                route = await self.get_semantic_route(message.message)
                return await self.route_message(route, message.message)

            except Exception as e:
                self.logger.exception("Chat processing failed", error=str(e))
                raise HTTPException(status_code=500, detail=str(e)) from e


    @property
    def router(self) -> APIRouter:
        """Return the underlying FastAPI router with registered endpoints."""
        return self._router

    async def get_semantic_route(self, message: str) -> SemanticRouterResponse:
        """
        Determine the semantic route for a message using AI provider.

        Args:
            message: Message to route

        Returns:
            SemanticRouterResponse: Determined route for the message
        """
        try:
            prompt, mime_type, schema = self.prompts.get_formatted_prompt(
                "semantic_router", user_input=message
            )
            route_response = self.ai.generate(
                prompt=prompt, response_mime_type=mime_type, response_schema=schema
            )
            return SemanticRouterResponse(route_response.text)
        except Exception as e:
            self.logger.exception("routing_failed", error=str(e))
            return SemanticRouterResponse.CONVERSATIONAL

    async def get_extraction_route(self, message: str) -> ExtractionPipelineResponse:
        """
        Determine the route for an extraction pipeline.

        Args:
            message: Message to route

        Returns:
            SemanticRouterResponse: Determined route for the message
        """
        return ExtractionPipelineResponse(message)

    async def route_extraction_pipeline(
        self, route: ExtractionPipelineResponse, message: str
    ) -> dict[str, str]:
        """
        Route a message to the appropriate handler based on semantic route.

        Args:
            route: Determined semantic route
            message: Original message to handle

        Returns:
            dict[str, str]: Response from the appropriate handler
        """
        # TODO: actually handle the routes by defining similar handle methods to the semantic router
        handlers = {
            ExtractionPipelineResponse.REQUEST_HTML_DATA: self.handle_html_extraction,
        }

        handler = handlers.get(route)
        if not handler:
            return {"response": "Unsupported route"}

        return await handler(message)

    async def handle_html_extraction(self, message: str) -> dict[str,str]:
        pipeline_json = json.loads(message);
        if "web_crawl" in pipeline_json:
            try:
                web_crawl_config = pipeline_json['web_crawl']
                for url in pipeline_json['web_crawl']['urls']:
                    # Make sure the front end actually sends this data lolol
                    crawl_webpage(url, web_crawl_config['use_llm'], web_crawl_config['max_pages'], web_crawl_config['class_grep']) 
                return {"response": "Succesfully crawled webpage"}
            except:
                self.logger.exception("Something went wrong with crawling webpage")
                return {"response": "Error in webpage crawling"}
        if "scrape" in pipeline_json: 
            try: 
                for url in pipeline_json['scrape']['urls']:
                    scrape_with_playwright(url, pipeline_json['scrape']['use_llm'])
                return {"response": "Sucessfully scraped webpage"}
            except:
                self.logger.exception("Something went wrong with scraping webpage")
                return {"response": "Error in webpage scraping"}
        return {"response": "No urls to crawl, did not run extraction"}

    async def route_message(
        self, route: SemanticRouterResponse, message: str
    ) -> dict[str, str]:
        """
        Route a message to the appropriate handler based on semantic route.

        Args:
            route: Determined semantic route
            message: Original message to handle

        Returns:
            dict[str, str]: Response from the appropriate handler
        """
        handlers = {
            SemanticRouterResponse.RAG_ROUTER: self.handle_rag_pipeline,
            SemanticRouterResponse.REQUEST_ATTESTATION: self.handle_attestation,
            SemanticRouterResponse.CONVERSATIONAL: self.handle_conversation,
        }

        handler = handlers.get(route)
        if not handler:
            return {"response": "Unsupported route"}

        return await handler(message)

    async def handle_rag_pipeline(self, _: str) -> dict[str, str]:
        """
        Handle attestation requests.

        Args:
            _: Unused message parameter

        Returns:
            dict[str, str]: Response containing attestation request
        """

        # Step 1. Classify the user query.
        prompt, mime_type, schema = self.prompts.get_formatted_prompt("rag_router")
        classification = self.query_router.route_query(
            prompt=prompt, response_mime_type=mime_type, response_schema=schema
        )
        self.logger.info("Query classified", classification=classification)

        if classification == "ANSWER":
            # Step 2. Retrieve relevant documents.
            retrieved_docs = self.retriever.semantic_search(_, top_k=5)
            self.logger.info("Documents retrieved")
            top_doc_name_score = {}
            for idx, doc in enumerate(retrieved_docs, start=1):
                identifier = doc.get("metadata", {}).get("filename", f"Doc{idx}")
                score = doc.get("score")
                top_doc_name_score[identifier] = score

            
            # Stringify the doc record
            record_string = json.dumps(top_doc_name_score)

            # Step 3. Generate the final answer.
            answer = self.responder.generate_response(_, retrieved_docs)
            self.logger.info("Response generated", answer=answer)
            return {"classification": classification, "response": answer, "doc_scores": record_string}

        # Map static responses for CLARIFY and REJECT.
        static_responses = {
            "CLARIFY": "Please provide additional context.",
            "REJECT": "The query is out of scope.",
        }

        if classification in static_responses:
            return {
                "classification": classification,
                "response": static_responses[classification],
            }

        self.logger.exception("RAG Routing failed")
        raise ValueError(classification)

    async def handle_attestation(self, _: str) -> dict[str, str]:
        """
        Handle attestation requests.

        Args:
            _: Unused message parameter

        Returns:
            dict[str, str]: Response containing attestation request
        """
        prompt = self.prompts.get_formatted_prompt("request_attestation")[0]
        request_attestation_response = self.ai.generate(prompt=prompt)
        self.attestation.attestation_requested = True
        return {"response": request_attestation_response.text}

    async def handle_conversation(self, message: str) -> dict[str, str]:
        """
        Handle general conversation messages.

        Args:
            message: Message to process

        Returns:
            dict[str, str]: Response from AI provider
        """
        response = self.ai.send_message(message)
        return {"response": response.text}
