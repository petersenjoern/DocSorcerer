"""Services for question and answer endpoint."""

from typing import List
from llama_index.response.schema import StreamingResponse
from llama_index.query_engine import RetrieverQueryEngine
from llama_index.indices.postprocessor.node import SimilarityPostprocessor
import weaviate
from ingestion.indexing import (CALLBACK_MANAGER, CONTEXT_WINDOW, EMBED_MODEL, 
                    MODEL_PATH, NODE_REFERENCES_PATH, NUM_OUTPUT, WEAVIATE_HOST, WEAVIATE_PORT, get_huggingface_llm, 
                    get_llama2, load_node_references, set_service_ctx, set_storage_ctx)

from llama_index import  ServiceContext, get_response_synthesizer, load_indices_from_storage
from llama_index.retrievers import VectorIndexRetriever
from llama_index.indices.base import BaseIndex
from llama_index.response_synthesizers.type import ResponseMode
from llama_index.retrievers import RecursiveRetriever


def run_query_with_engine(query_engine:  RetrieverQueryEngine, question: str) -> StreamingResponse:
    """Run query engines and return answer"""

    return query_engine.query(question)

# TODO: decouple connections 
def load_indices_and_query_engine() -> RetrieverQueryEngine:
    """Load llamaindex service and storage context, intiate query engine"""

    llm = get_llama2(
        model_path=MODEL_PATH,
        max_new_tokens=NUM_OUTPUT,
        model_temperature=0.1,
        context_window=CONTEXT_WINDOW
    )

    # llm = get_huggingface_llm(
    #     model_name="Writer/camel-5b-hf",
    #     max_new_tokens=NUM_OUTPUT,
    #     model_temperature=0.1,
    #     context_window=CONTEXT_WINDOW
    # )

    service_context = set_service_ctx(llm=llm, embed_model=EMBED_MODEL, callback_manager=CALLBACK_MANAGER)
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


    nodes_indexed = load_node_references(NODE_REFERENCES_PATH)
    all_node_dict = {n.node_id: n for n in nodes_indexed}
    retriever_metadata = RecursiveRetriever(
        "vector",
        retriever_dict={"vector": load_vector_index_retriever(service_context=service_context, storage_indices=storage_indices)},
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



def load_vector_index_retriever(service_context: ServiceContext, storage_indices: List[BaseIndex]) -> VectorIndexRetriever:
    """Loading vector index retriever"""
    
    return VectorIndexRetriever(
        index=storage_indices[0],
        similarity_top_k=2,
        vector_store_query_mode="default", # 'hybrid' only working with weaviate
        alpha=0.8, # alpha = 0 -> bm25, alpha=1 -> vector search, only working with hybrid
        service_context=service_context
    )
