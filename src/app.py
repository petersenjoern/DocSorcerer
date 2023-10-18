"""App to Q&A/ Chat with documents."""

from typing import AsyncGenerator, List
import uvicorn
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from langchain.embeddings import HuggingFaceEmbeddings
from llama_index import LangchainEmbedding, ServiceContext, get_response_synthesizer, load_indices_from_storage
from llama_index.retrievers import VectorIndexRetriever
from llama_index.indices.base import BaseIndex

from llama_index.callbacks import (CallbackManager,LlamaDebugHandler)
from llama_index.response_synthesizers.type import ResponseMode
from llama_index.query_engine import RetrieverQueryEngine
from llama_index.indices.postprocessor.node import SimilarityPostprocessor
import weaviate
from indexing import (CONTEXT_WINDOW, EMBED_MODEL_NAME, 
                    MODEL_PATH, NUM_OUTPUT, WEAVIATE_HOST, WEAVIATE_PORT, 
                    get_llama2, set_service_ctx, set_storage_ctx)


def load_indices_and_query_engine():
    """Load llamaindex service and storage context, intiate query engine"""

    service_context = load_service_context()
    weaviate_client = weaviate.Client(url=f"http://{WEAVIATE_HOST}:{WEAVIATE_PORT}")
    storage_context = set_storage_ctx(weaviate_client)
    storage_indices = load_indices_from_storage(storage_context=storage_context, service_context=service_context)


    
    # configure response synthesizer
    response_synthesizer = get_response_synthesizer(
        service_context=service_context,
        streaming=True,
        use_async=True,
        response_mode=ResponseMode.COMPACT # alternative: TREE_SUMMARIZE
    )

    # assemble query engine
    query_engine = RetrieverQueryEngine(
        retriever=load_vector_index_retriever(service_context=service_context, storage_indices=storage_indices),
        response_synthesizer=response_synthesizer,
        node_postprocessors=[
            SimilarityPostprocessor(similarity_cutoff=0.75)
        ]
    )
    return query_engine

def load_service_context() -> ServiceContext:
    """Loading the service context"""
    
    llama_debug = LlamaDebugHandler(print_trace_on_end=True)
    callback_manager = CallbackManager([llama_debug])

    llm = get_llama2(
        model_path=MODEL_PATH,
        max_new_tokens=NUM_OUTPUT,
        model_temperature=0.1,
        context_window=CONTEXT_WINDOW
    )

    embed_model = LangchainEmbedding(HuggingFaceEmbeddings(
            model_name=EMBED_MODEL_NAME,
            model_kwargs={"device": "cuda"},
            encode_kwargs={'normalize_embeddings': False}
        ))

    return set_service_ctx(llm=llm, embed_model=embed_model, callback_manager=callback_manager)

def load_vector_index_retriever(service_context: ServiceContext, storage_indices: List[BaseIndex]) -> VectorIndexRetriever:
    """Loading vector index retriever"""
    
    return VectorIndexRetriever(
        index=storage_indices[0],
        similarity_top_k=3,
        vector_store_query_mode="default", # 'hybrid' only working with weaviate
        alpha=0.8, # alpha = 0 -> bm25, alpha=1 -> vector search, only working with hybrid
        service_context=service_context
    )


query_engines = {}

@asynccontextmanager
async def lifespan(app: FastAPI):
    """FastApi lifespan - code that is executed before the application starts up."""

    query_engines["vector_indices_and_query_engine"] = load_indices_and_query_engine()
    yield
    # Clean up and release resources
    query_engines.clear()

app = FastAPI(lifespan=lifespan)

def run_query_engine(question: str) -> AsyncGenerator:
    """Run query engine with question"""

    query_engine = query_engines["vector_indices_and_query_engine"]
    response_iter = query_engine.query(question)

    for text in response_iter.response_gen:
        yield f"{text}"
    yield f"\n\n"

    # finall return source node information.
    source_filename = [n.node.metadata["file_name"] for n in response_iter.source_nodes]
    source_page = [n.node.metadata["page_label"] for n in response_iter.source_nodes]
    source_relevance_score = [str(round(n.score,2)) for n in response_iter.source_nodes]
    source_data = list(zip(source_filename, source_page, source_relevance_score))
    source_data_str = [" ".join(tup) for tup in source_data]
    yield f"Supporting evidence: \n\n"
    for evidence in source_data_str:
        yield f"{evidence}\n\n"

@app.get("/ask")
async def ask_documents(question: str) -> StreamingResponse:
    """API endpoint to stream query responses"""

    return StreamingResponse(run_query_engine(question), media_type="text/event-stream")


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
