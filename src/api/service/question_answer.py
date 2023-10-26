"""Services for question and answer endpoint."""

from typing import AsyncGenerator, List, Union

import repository.question_answer as repository_question_answer
from dto.node import NodeWithEvidence
from llama_index.indices.base_retriever import BaseRetriever
from llama_index.query_engine import RetrieverQueryEngine
from llama_index.response.schema import StreamingResponse
from llama_index.retrievers import VectorIndexRetriever
from llama_index.schema import NodeWithScore


def evidence_for_answer(
    retriever: Union[VectorIndexRetriever, BaseRetriever], question: str
) -> List[NodeWithEvidence]:
    """Return source node information for question asked"""

    source_nodes = repository_question_answer.retriever_get_nodes(retriever, question)
    return _format_response_source_nodes(source_nodes)


def answer_question(
    query_engine: RetrieverQueryEngine, question: str
) -> AsyncGenerator:
    """Answer a question by calling the"""

    response_iter = repository_question_answer.engine_run_query_with_question(
        query_engine, question
    )
    return _stream_response(response_iter)


def _stream_response(response_iter: StreamingResponse) -> AsyncGenerator:
    """Run query engine with question"""

    for text in response_iter.response_gen:
        yield f"{text}"
    yield "\n\n"


def _format_response_source_nodes(
    source_nodes: List[NodeWithScore]
) -> List[NodeWithEvidence]:
    """Format source node information to only return what is desired"""

    return [
        NodeWithEvidence(node_id=s.node_id, score=round(s.score, 3), text=s.text)
        for s in source_nodes
    ]
