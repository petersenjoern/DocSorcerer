"""Services for question and answer endpoint."""

from typing import AsyncGenerator, List
from llama_index.response.schema import StreamingResponse
from llama_index.schema import NodeWithScore

def stream_response(response_iter: StreamingResponse) -> AsyncGenerator:
    """Run query engine with question"""

    for text in response_iter.response_gen:
        yield f"{text}"
    yield f"\n\n"

    # finall return source node information.
    source_data_str = _format_response_source_nodes(response_iter.source_nodes)
    yield f"Supporting evidence: \n\n"
    for evidence in source_data_str:
        yield f"{evidence}\n\n"


def _format_response_source_nodes(source_nodes: List[NodeWithScore]) -> List[str]:
    """Format source node information, so it can be streamed as strings"""

    source_node_ids = [n.node_id for n in source_nodes]
    source_text = [n.text for n in source_nodes]
    source_relevance_score = [str(round(n.score,2)) for n in source_nodes]
    source_data = list(zip(source_node_ids, source_text, source_relevance_score))
    return [" ".join(tup) for tup in source_data]