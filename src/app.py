"""App to Q&A/ Chat with documents."""


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


if __name__ == "__main__":

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
        response_mode=ResponseMode.COMPACT # alternatives: COMPACT
    )

    # assemble query engine
    query_engine = RetrieverQueryEngine(
        retriever=vector_index_retriever,
        response_synthesizer=response_synthesizer,
        node_postprocessors=[
            SimilarityPostprocessor(similarity_cutoff=0.7)
        ]
    )


response = query_engine.query("what is the research paper about?")
response.print_response_stream()
