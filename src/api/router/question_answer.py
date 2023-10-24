"""Router for question and answer endpoint"""

from typing import List
import weaviate

from contextlib import asynccontextmanager
from fastapi import FastAPI
from llama_index import PromptHelper, ServiceContext, load_indices_from_storage
from llama_index.response.schema import StreamingResponse
from dto.node import NodeWithEvidence
from indexing import CHUNK_SIZE, EMBED_MODEL
from language_models import get_llama2
from llamaindex_storage import set_storage_ctx
from llama_index.callbacks import (CallbackManager,LlamaDebugHandler)
from query_engine import initialise_query_engine
from llama_index.retrievers import VectorIndexRetriever
from llama_index.schema import NodeWithScore

from fastapi import APIRouter
from fastapi.responses import StreamingResponse
import service.question_answer as service_question_answer

router = APIRouter()


LLAMA_INDEX_CALLBACKS_API = CallbackManager([LlamaDebugHandler(print_trace_on_end=True)])

# Weaviate
WEAVIATE_HOST = "localhost"
WEAVIATE_PORT = 8080

# LLM
CONTEXT_WINDOW_API = 3500
NUM_OUTPUT_API = 596
# Prompt Helper - can help deal with LLM context window token limitations
PROMPT_HELPER_API = PromptHelper(
    context_window=CONTEXT_WINDOW_API,
    num_output=NUM_OUTPUT_API,
    chunk_overlap_ratio=0.1,
    chunk_size_limit=CHUNK_SIZE
)

lifespan_objects = {}

@asynccontextmanager
async def lifespan(app: FastAPI):
    """FastApi lifespan - code that is executed before the application starts up."""

    llm = get_llama2(
        max_new_tokens=NUM_OUTPUT_API,
        model_temperature=0.1,
        context_window=CONTEXT_WINDOW_API
    )
    service_context = ServiceContext.from_defaults(
        llm=llm,
        chunk_size=CHUNK_SIZE,
        callback_manager=LLAMA_INDEX_CALLBACKS_API,
        embed_model=EMBED_MODEL,
        prompt_helper=PROMPT_HELPER_API
    )

    weaviate_client = weaviate.Client(url=f"http://{WEAVIATE_HOST}:{WEAVIATE_PORT}")
    storage_context = set_storage_ctx(weaviate_client)
    storage_indices = load_indices_from_storage(storage_context=storage_context, service_context=service_context)

    vector_index_retriever=VectorIndexRetriever(
        index=storage_indices[0],
        similarity_top_k=2,
        vector_store_query_mode="default", # 'hybrid' only working with weaviate
        alpha=0.8, # alpha = 0 -> bm25, alpha=1 -> vector search, only working with hybrid
        service_context=service_context
    )
    query_engine = initialise_query_engine(service_context, vector_index_retriever)

    lifespan_objects["llm"] = llm
    lifespan_objects["query_engine"] = query_engine
    lifespan_objects["retriever"] = vector_index_retriever
    yield
    # Clean up and release resources
    lifespan_objects.clear()


@router.get("/answer-question")
async def answer_question_with_documents(question: str) -> StreamingResponse:
    """API endpoint to stream llm response for question asked"""

    query_engine = lifespan_objects["query_engine"]
    return StreamingResponse(service_question_answer.answer_question(query_engine, question), media_type="text/event-stream")

# TODO: change NodeWithEvidence to Response Object
@router.get("/answer-question-evidence")
async def evidence_for_question(question: str) -> List[NodeWithEvidence]:
    """API endpoint to stream query responses"""

    retriever = lifespan_objects["retriever"]
    return service_question_answer.evidence_for_answer(retriever, question)
