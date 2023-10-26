"""Router for question and answer endpoint"""

from contextlib import asynccontextmanager
from typing import List

import service.question_answer as service_question_answer
import weaviate
from dto.node import NodeWithEvidence
from fastapi import APIRouter, FastAPI
from fastapi.responses import StreamingResponse
from llama_index import PromptHelper, ServiceContext, load_indices_from_storage
from llama_index.callbacks import CallbackManager, LlamaDebugHandler
from llama_index.retrievers import VectorIndexRetriever
from query_engine import initialise_query_engine

from config import get_api_settings
from ingestion.indexing import EMBED_MODEL
from models.language_models import get_llama2
from storage.llamaindex_storage import set_storage_ctx

LLAMA_INDEX_CALLBACKS_API = CallbackManager(
    [LlamaDebugHandler(print_trace_on_end=True)]
)

router = APIRouter()

lifespan_objects = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    """FastApi lifespan - code that is executed before the application starts up."""

    settings = get_api_settings()

    llm = get_llama2(
        max_new_tokens=settings.llm.num_output,
        model_temperature=settings.llm.temperature,
        context_window=settings.llm.context_window,
    )
    service_context = ServiceContext.from_defaults(
        llm=llm,
        chunk_size=settings.parser.chunk_size,
        callback_manager=LLAMA_INDEX_CALLBACKS_API,
        embed_model=EMBED_MODEL,
        prompt_helper=PromptHelper(
            context_window=settings.prompt_helper.context_window,
            num_output=settings.prompt_helper.num_output,
            chunk_overlap_ratio=0.1,
            chunk_size_limit=settings.parser.chunk_size,
        ),
    )

    weaviate_client = weaviate.Client(
        url=f"http://{settings.db_vector.host}:{settings.db_vector.port}"
    )
    storage_context = set_storage_ctx(weaviate_client)
    storage_indices = load_indices_from_storage(
        storage_context=storage_context, service_context=service_context
    )

    vector_index_retriever = VectorIndexRetriever(
        index=storage_indices[0],
        similarity_top_k=2,
        vector_store_query_mode="default",  # 'hybrid' only working with weaviate
        alpha=0.8,  # alpha = 0 -> bm25, alpha=1 -> vector search, only working with hybrid
        service_context=service_context,
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
    return StreamingResponse(
        service_question_answer.answer_question(query_engine, question),
        media_type="text/event-stream",
    )


# TODO: change NodeWithEvidence to Response Object
@router.get("/answer-question-evidence")
async def evidence_for_question(question: str) -> List[NodeWithEvidence]:
    """API endpoint to stream query responses"""

    retriever = lifespan_objects["retriever"]
    return service_question_answer.evidence_for_answer(retriever, question)
