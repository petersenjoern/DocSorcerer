# Build and query index

import logging
import sys
from pathlib import Path
from typing import Dict, List, Type

import weaviate
from langchain.embeddings import HuggingFaceEmbeddings
from llama_hub.file.image.base import ImageReader as ImageReaderForFlatPDF
from utils.pdf import PDFReaderCustom

from llama_index import (KeywordTableIndex, ServiceContext,
                         SimpleDirectoryReader, VectorStoreIndex,
                         download_loader, get_response_synthesizer)
from llama_index.callbacks import (CallbackManager, CBEventType,
                                   LlamaDebugHandler)
from llama_index.embeddings.langchain import LangchainEmbedding
from llama_index.indices.document_summary import (
    DocumentSummaryIndex, DocumentSummaryIndexRetriever)
from llama_index.indices.document_summary.retrievers import \
    DocumentSummaryIndexLLMRetriever
from llama_index.indices.loading import (load_index_from_storage,
                                         load_indices_from_storage)
from llama_index.indices.postprocessor.node import SimilarityPostprocessor
from llama_index.indices.prompt_helper import PromptHelper
from llama_index.llms import LlamaCPP
from llama_index.llms.base import LLM
from llama_index.llms.llama_utils import (completion_to_prompt,
                                          messages_to_prompt)
from llama_index.node_parser import SimpleNodeParser
from llama_index.node_parser.extractors import (EntityExtractor,
                                                KeywordExtractor,
                                                MetadataExtractor,
                                                QuestionsAnsweredExtractor,
                                                SummaryExtractor,
                                                TitleExtractor)
from llama_index.node_parser.extractors.metadata_extractors import \
    DEFAULT_ENTITY_MODEL
from llama_index.query_engine import RetrieverQueryEngine
from llama_index.query_engine.router_query_engine import RouterQueryEngine
from llama_index.readers.base import BaseReader
from llama_index.readers.file.image_reader import ImageReader
from llama_index.response_synthesizers.type import ResponseMode
from llama_index.retrievers import VectorIndexRetriever
from llama_index.schema import Document, TextNode
from llama_index.selectors.llm_selectors import LLMSingleSelector
from llama_index.selectors.prompts import DEFAULT_SINGLE_SELECT_PROMPT
from llama_index.storage.docstore import MongoDocumentStore
from llama_index.storage.index_store import MongoIndexStore
from llama_index.storage.storage_context import StorageContext
from llama_index.text_splitter.token_splitter import TokenTextSplitter
from llama_index.tools.query_engine import QueryEngineTool
from llama_index.vector_stores.types import ExactMatchFilter, MetadataFilters
from llama_index.vector_stores.weaviate import WeaviateVectorStore

MODEL_NAME = "llama-2-13b-chat.gguf"
MODEL_PATH = str(Path.cwd().joinpath("models", MODEL_NAME))
CONTEXT_WINDOW = 3800
NUM_OUTPUT = 256
CHUNK_SIZE = 400
DATA_PATH = str(Path.cwd().joinpath("data", "LessIsMoreForAlignment"))
INDEX_PATH = str(Path.cwd().joinpath("indexes", "testing"))
WEAVIATE_INDEX_NAME = "Vectors"
MONGO_DB_NAME = "Stores"
MONGO_INDEX_STORE_NAMESPACE = "IndexStore"
MONGO_DOC_STORE_NAMESPACE = "DocStore"
RETRAIN = True
VECTOR_RETRIEVER = True


logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)
logging.getLogger().addHandler(logging.StreamHandler(stream=sys.stdout))


FILE_READER_CLS: Dict[str, Type[BaseReader]] = {
    ".pdf": PDFReaderCustom(image_loader=ImageReaderForFlatPDF(text_type="plain_text")),
    ".jpg": ImageReader,
    ".png": ImageReader,
    ".jpeg": ImageReader
}


def get_llama2(model_temperature: int=0.1, context_window:int=3800):
    """Init llama-cpp-python https://github.com/abetlen/llama-cpp-python via langchain.llm"""
    
    # llama2 has a context window of 4096 tokens
    return LlamaCPP(
        model_path=MODEL_PATH,
        context_window=context_window,
        temperature=model_temperature,
        max_new_tokens=NUM_OUTPUT,
        model_kwargs={"n_gpu_layers": 50, "n_batch": 8, "use_mlock": False},
        messages_to_prompt=messages_to_prompt,
        completion_to_prompt=completion_to_prompt,
        verbose=True)

def load_data():
    """Custom loading data into llama index documents class"""
    documents = SimpleDirectoryReader(
        input_dir=DATA_PATH,
        filename_as_id=True,
        recursive=True,
        file_extractor=FILE_READER_CLS).load_data()
    return documents

def inserts():
    pass

def records_to_insert(inmemory_records: List[str], stored_records: List[str]) -> List[str]:
    """Records not stored, to be inserted."""
    return list(set(inmemory_records) - set(stored_records))

def records_to_update(stored_records: List[str], inmemory_records: List[str]) -> List[str]:
    """potential records to update."""
    return list(set(stored_records).intersection(inmemory_records))

def update(documents: List[Document]) -> None:
    """update records in vector store.

    1. delete records in vector store.
    2. insert records in vector store.
    """
    nodes = llama_index_preprocessing(documents)
    pass


def deletes():
    pass

def llama_index_preprocessing(documents: List[Document], llm: LLM) -> List[TextNode]:
    """Process (meta) data before indexing"""

    text_splitter=TokenTextSplitter(separator=" ", chunk_size=CHUNK_SIZE, chunk_overlap=20)
    # text_chunks = text_splitter.split_text(documents.text)
    # doc_chunks = [Document(text=t, id=f"{doc_id_{i}}") for i,t in enumerate(text_chunks)]

    metadata_extractor = MetadataExtractor(
        extractors=[
            #TitleExtractor(nodes=5, llm=llm),
            #QuestionsAnsweredExtractor(questions=2, llm=llm),
            #SummaryExtractor(summaries=["self"], llm=llm),
            #KeywordExtractor(keywords=4, llm=llm),
            #EntityExtractor(prediction_threshold=0.5, model_name=DEFAULT_ENTITY_MODEL, device="cuda"),
        ],
        in_place=True,
    )

    node_parser = SimpleNodeParser.from_defaults(
        text_splitter=text_splitter,
        #metadata_extractor=metadata_extractor,
        include_metadata=True,
        include_prev_next_rel=False
    )

    nodes = node_parser.get_nodes_from_documents(documents=documents, show_progress=True)
    return nodes

def get_doc_ids_in_store(client: weaviate.Client, class_name: str) -> List[str]:
    """Return all doc ids in weaviate store"""

    try:
        client.schema.get(class_name)
    except weaviate.UnexpectedStatusCodeException:
        return []

    all_objects_store = client.data_object.get(class_name=class_name)
    all_doc_ids_store = [i["properties"]["doc_id"] for i in all_objects_store["objects"]]
    return all_doc_ids_store


def purge_vector_store(client: weaviate.Client, class_name: str):
    """Purge vector store for a schema (class name)"""
    try:
        client.schema.delete_class(class_name)
    except:
        pass

def filter_documents_by_doc_ids(documents: List[Document], doc_ids: List[str]) -> List[Document]:
    """Filter documents by doc ids"""

    result = filter(lambda x: x.doc_id in doc_ids, documents)
    return list(result)


if __name__ == "__main__":

    # llm defintion
    llama_debug = LlamaDebugHandler(print_trace_on_end=True)
    callback_manager = CallbackManager([llama_debug])

    llm = get_llama2(context_window=CONTEXT_WINDOW)
    service_context = ServiceContext.from_defaults(
        llm=llm,
        chunk_size=CHUNK_SIZE,
        callback_manager=callback_manager,
        embed_model=LangchainEmbedding(HuggingFaceEmbeddings(
            model_name="BAAI/bge-small-en",
            model_kwargs={"device": "cuda"},
            encode_kwargs={'normalize_embeddings': False}
            )
        ),
        prompt_helper=PromptHelper(
            context_window=CONTEXT_WINDOW,
            num_output=NUM_OUTPUT,
            chunk_overlap_ratio=0.2,
            chunk_size_limit=CHUNK_SIZE
        )
    )


    # embeddings and docs are stored within a Weaviate collection
    weaviate_client = weaviate.Client("http://localhost:8080")
    vector_store = WeaviateVectorStore(weaviate_client=weaviate_client, index_name=WEAVIATE_INDEX_NAME)
    # client.schema.get()
    # client.schema.delete_class("class_name")
    # client.schema.delete_all()


    # index store
    index_store = MongoIndexStore.from_host_and_port(
        host="localhost",
        port=27017,
        db_name=MONGO_DB_NAME,
        namespace=MONGO_INDEX_STORE_NAMESPACE
    )

    # doc store
    doc_store = MongoDocumentStore.from_host_and_port(
        host="localhost",
        port=27017,
        db_name=MONGO_DB_NAME,
        namespace=MONGO_DOC_STORE_NAMESPACE
    )

    # load storage context and index
    storage_context = StorageContext.from_defaults(vector_store=vector_store, index_store=index_store, docstore=doc_store)

    #index = VectorStoreIndex.from_vector_store(vector_store=vector_store, service_context=service_context)
    storage_indices = load_indices_from_storage(storage_context=storage_context, service_context=service_context)



    if RETRAIN:
        documents = load_data()
        purge_vector_store(weaviate_client, WEAVIATE_INDEX_NAME)
        all_doc_ids_memory = [document.doc_id for document in documents]
        all_doc_ids_store = get_doc_ids_in_store(weaviate_client, WEAVIATE_INDEX_NAME)
        doc_ids_to_insert = records_to_insert(all_doc_ids_memory, all_doc_ids_store)
        docs_to_insert = filter_documents_by_doc_ids(documents, doc_ids_to_insert)
        #nodes_to_insert=llama_index_preprocessing(docs_to_insert, llm=llm)

        # create (or load) docstore and add nodes
        # storage_context.docstore.add_documents(nodes)

        # build index
        response_synthesizer = get_response_synthesizer(
            response_mode="tree_summarize", service_context=service_context, use_async=True
        )
        storage_indices = []
        vector_store_index = VectorStoreIndex.from_documents(
            documents=docs_to_insert,
            storage_context=storage_context,
            service_context=service_context,
            response_synthesizer=response_synthesizer,
            show_progress=True
        )
        storage_indices.extend([vector_store_index])

        # The indexing is working, however, in the retrieval with Llama2 is a bug, therefore not usable right now
        #doc_summary_index = DocumentSummaryIndex.from_documents(
        #    documents=docs_to_insert,
        #    service_context=service_context,
        #    storage_context=storage_context,
        #    response_synthesizer=response_synthesizer,
        #    show_progress=True
        #)
        #storage_indices.extend([doc_summary_index])




    # configure retriever
    if VECTOR_RETRIEVER:
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

        #vector_tool = QueryEngineTool.from_defaults(
        #    query_engine=storage_indices[0].as_query_engine(),
        #    description="Useful for retrieving specific context.",
        #)
#
        #query_engine = RouterQueryEngine(
        #    selector=LLMSingleSelector.from_defaults(service_context=service_context),
        #    query_engine_tools=[vector_tool],
        #    service_context=service_context
        #)
    else:
        # isnt working right now, there is a bug in the llama2 retriever for the document summary index
        document_summary_retriever = DocumentSummaryIndexLLMRetriever(
            index=storage_indices[0],
            service_context=service_context
            # choice_select_prompt=choice_select_prompt,
            # choice_batch_size=choice_batch_size,
            # format_node_batch_fn=format_node_batch_fn,
            # parse_choice_select_answer_fn=parse_choice_select_answer_fn,
        )
        response_synthesizer = get_response_synthesizer(response_mode="tree_summarize", service_context=service_context)
        # assemble query engine
        query_engine = RetrieverQueryEngine(
            retriever=document_summary_retriever,
            response_synthesizer=response_synthesizer,
        )

    #query_engine = index.as_query_engine(response_mode="tree_summarize")
    #query_engine = index.as_query_engine(
    #    similarity_top_k=3,
    #    response_mode="tree_summarize",
    #    vector_store_query_mode="default",
    #    filters=None,
    #    alpha=None,
    #    doc_ids=None,
    #)
    response = query_engine.query("what is the research paper about?")
    response.print_response_stream()

    ## chat mode; not working with llama2
    #chat_engine = storage_indices[0].as_chat_engine(verbose=True, service_context=service_context)
    #response = chat_engine.chat("What is an LLM?")
    #print(response)

