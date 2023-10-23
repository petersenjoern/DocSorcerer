"""Services for question and answer endpoint."""

from llama_index.response.schema import StreamingResponse
from llama_index.query_engine import RetrieverQueryEngine
from llama_index.indices.postprocessor.node import SimilarityPostprocessor
import weaviate
from ingestion.indexing import (CHUNK_SIZE, LLAMA_INDEX_CALLBACKS, EMBED_MODEL, NODE_REFERENCES_PATH, load_node_references)
from llama_index import  PromptHelper, ServiceContext, StorageContext, get_response_synthesizer, load_indices_from_storage
from llama_index.retrievers import VectorIndexRetriever
from llama_index.response_synthesizers.type import ResponseMode
from llama_index.retrievers import RecursiveRetriever
from llamaindex_storage import set_storage_ctx
from models.language_models import get_llama2

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

def run_query_with_engine(query_engine:  RetrieverQueryEngine, question: str) -> StreamingResponse:
    """Run query engines and return answer"""

    return query_engine.query(question)


# TODO: decouple connections 
def load_indices_and_query_engine() -> RetrieverQueryEngine:
    """Load llamaindex service and storage context, intiate query engine"""

    llm = get_llama2(
        max_new_tokens=NUM_OUTPUT_API,
        model_temperature=0.1,
        context_window=CONTEXT_WINDOW_API
    )

    service_context = ServiceContext.from_defaults(
        llm=llm,
        chunk_size=CHUNK_SIZE,
        callback_manager=LLAMA_INDEX_CALLBACKS,
        embed_model=EMBED_MODEL,
        prompt_helper=PROMPT_HELPER_API
    )
    weaviate_client = weaviate.Client(url=f"http://{WEAVIATE_HOST}:{WEAVIATE_PORT}")
    storage_context = set_storage_ctx(weaviate_client)


    # configure response synthesizer
    response_synthesizer = get_response_synthesizer(
        service_context=service_context,
        streaming=True,
        use_async=True,
        response_mode=ResponseMode.COMPACT # alternative: TREE_SUMMARIZE
    )


    nodes_indexed = load_node_references(NODE_REFERENCES_PATH)
    all_node_dict = {n.node_id: n for n in nodes_indexed}
    retriever_metadata = RecursiveRetriever(
        "vector",
        retriever_dict={"vector": load_vector_index_retriever(service_context=service_context, storage_context=storage_context)},
        node_dict=all_node_dict,
        verbose=True,
    )


    query_engine = RetrieverQueryEngine.from_args(
        retriever=retriever_metadata,
        service_context=service_context,
        response_synthesizer=response_synthesizer,
        node_postprocessors=[
            SimilarityPostprocessor(similarity_cutoff=0.75)
        ],
        verbose=True,
        streaming=True,
        use_async=True,
    )

    return query_engine



def load_vector_index_retriever(service_context: ServiceContext, storage_context: StorageContext) -> VectorIndexRetriever:
    """Loading vector index retriever"""
    
    storage_indices = load_indices_from_storage(storage_context=storage_context, service_context=service_context)

    return VectorIndexRetriever(
        index=storage_indices[0],
        similarity_top_k=2,
        vector_store_query_mode="default", # 'hybrid' only working with weaviate
        alpha=0.8, # alpha = 0 -> bm25, alpha=1 -> vector search, only working with hybrid
        service_context=service_context
    )
