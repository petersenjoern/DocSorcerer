"""App to Q&A/ Chat with documents."""

from typing import AsyncGenerator
import uvicorn
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from langchain.embeddings import HuggingFaceEmbeddings
from llama_index import LangchainEmbedding, get_response_synthesizer, load_indices_from_storage
from llama_index.retrievers import VectorIndexRetriever
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

    service_context = set_service_ctx(llm=llm, embed_model=embed_model, callback_manager=callback_manager)
    weaviate_client = weaviate.Client(url=f"http://{WEAVIATE_HOST}:{WEAVIATE_PORT}")
    storage_context = set_storage_ctx(weaviate_client)
    storage_indices = load_indices_from_storage(storage_context=storage_context, service_context=service_context)

    vector_index_retriever = VectorIndexRetriever(
        index=storage_indices[0],
        similarity_top_k=3,
        vector_store_query_mode="default", # 'hybrid' only working with weaviate
        alpha=0.8, # alpha = 0 -> bm25, alpha=1 -> vector search, only working with hybrid
        service_context=service_context
    )

    # configure response synthesizer
    response_synthesizer = get_response_synthesizer(
        service_context=service_context,
        streaming=True,
        use_async=True,
        response_mode=ResponseMode.COMPACT
    )

    # assemble query engine
    query_engine = RetrieverQueryEngine(
        retriever=vector_index_retriever,
        response_synthesizer=response_synthesizer,
        node_postprocessors=[
            SimilarityPostprocessor(similarity_cutoff=0.7)
        ]
    )
    return query_engine


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

@app.get("/ask")
async def question_documents(question: str) -> StreamingResponse:
    """API endpoint to stream query responses"""

    return StreamingResponse(run_query_engine(question), media_type="text/event-stream")


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
