"""Query Engine available to the API"""

from ingestion.indexing import NODE_REFERENCES_PATH, _load_node_references
from llama_index import ServiceContext, get_response_synthesizer
from llama_index.indices.postprocessor.node import SimilarityPostprocessor
from llama_index.query_engine import RetrieverQueryEngine
from llama_index.response_synthesizers.type import ResponseMode
from llama_index.retrievers import RecursiveRetriever, VectorIndexRetriever


def initialise_query_engine(
    service_context: ServiceContext, vector_retriever: VectorIndexRetriever
) -> RetrieverQueryEngine:
    """Load llamaindex service and storage context, intiate query engine"""

    # configure response synthesizer
    response_synthesizer = get_response_synthesizer(
        service_context=service_context,
        streaming=True,
        use_async=True,
        response_mode=ResponseMode.COMPACT,  # alternative: TREE_SUMMARIZE
    )

    nodes_indexed = _load_node_references(NODE_REFERENCES_PATH)
    all_node_dict = {n.node_id: n for n in nodes_indexed}
    retriever_metadata = RecursiveRetriever(
        "vector",
        retriever_dict={"vector": vector_retriever},
        node_dict=all_node_dict,
        verbose=True,
    )

    query_engine = RetrieverQueryEngine.from_args(
        retriever=retriever_metadata,
        service_context=service_context,
        response_synthesizer=response_synthesizer,
        node_postprocessors=[SimilarityPostprocessor(similarity_cutoff=0.70)],
        verbose=True,
        streaming=True,
        use_async=True,
    )

    return query_engine
