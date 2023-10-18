"""Router for question and answer endpoint"""

from contextlib import asynccontextmanager
from fastapi import APIRouter, FastAPI
from fastapi.responses import StreamingResponse
import repository.question_answer as repository_question_answer
import service.question_answer as service_question_answer

router = APIRouter()

query_engines = {}

@asynccontextmanager
async def lifespan(app: FastAPI):
    """FastApi lifespan - code that is executed before the application starts up."""

    query_engines["vector_indices_and_query_engine"] = repository_question_answer.load_indices_and_query_engine()
    yield
    # Clean up and release resources
    query_engines.clear()


@router.get("/ask")
async def ask_documents(question: str) -> StreamingResponse:
    """API endpoint to stream query responses"""

    query_engine = query_engines["vector_indices_and_query_engine"]
    response_iter = repository_question_answer.run_query_with_engine(query_engine, question)
    return StreamingResponse(service_question_answer.stream_response(response_iter), media_type="text/event-stream")
