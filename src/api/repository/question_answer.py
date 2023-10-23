"""Services for question and answer endpoint."""

from llama_index.response.schema import StreamingResponse
from llama_index.query_engine import RetrieverQueryEngine


def engine_run_query_with_question(query_engine:  RetrieverQueryEngine, question: str) -> StreamingResponse:
    """Run query engines and return answer"""

    return query_engine.query(question)
