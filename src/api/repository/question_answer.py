"""Services for question and answer endpoint."""

from typing import List, Union
from llama_index.response.schema import StreamingResponse
from llama_index.query_engine import RetrieverQueryEngine
from llama_index.indices.base_retriever import BaseRetriever
from llama_index.retrievers import VectorIndexRetriever
from llama_index.schema import NodeWithScore



def engine_run_query_with_question(query_engine:  RetrieverQueryEngine, question: str) -> StreamingResponse:
    """Run query engines and return answer"""

    return query_engine.query(question)


def retriever_get_nodes(retriever: Union[VectorIndexRetriever, BaseRetriever], question: str) -> List[NodeWithScore]:
    """Retriever get NodeWithScore for question/query"""

    return retriever.retrieve(question)