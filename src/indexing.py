"""Indexing documents with llamaindex"""

import logging
import sys
from pathlib import Path
from typing import Dict, List, Type

import weaviate
from langchain.embeddings import HuggingFaceEmbeddings
from llama_hub.file.image.base import ImageReader as ImageReaderForFlatPDF
from utils.pdf import PDFReaderCustom

from llama_index.embeddings.utils import EmbedType
from llama_index.llms.custom import CustomLLM
from llama_index import (ServiceContext, SimpleDirectoryReader, SummaryIndex, VectorStoreIndex, get_response_synthesizer)
from llama_index.callbacks import (CallbackManager,LlamaDebugHandler)
from llama_index.embeddings.langchain import LangchainEmbedding
from llama_index.indices.loading import load_indices_from_storage
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
from llama_index.readers.base import BaseReader
from llama_index.readers.file.image_reader import ImageReader
from llama_index.readers.file.docs_reader import DocxReader
from llama_index.schema import Document, TextNode
from llama_index.storage.docstore import MongoDocumentStore
from llama_index.storage.index_store import MongoIndexStore
from llama_index.storage.storage_context import StorageContext
from llama_index.text_splitter.token_splitter import TokenTextSplitter
from llama_index.vector_stores.weaviate import WeaviateVectorStore

# TODO: move most of these to a cfg file
# LLM Model CFG
MODEL_NAME = "llama-2-13b-chat.gguf"
MODEL_PATH = str(Path.cwd().joinpath("models", MODEL_NAME))
CONTEXT_WINDOW = 3800
NUM_OUTPUT = 300
CHUNK_SIZE = 1024

# Embedding Model
EMBED_MODEL_NAME = "BAAI/bge-small-en"

# Data
DATA_PATH = str(Path.cwd().joinpath("data", "Productivity"))

# Weaviate
WEAVIATE_HOST = "localhost"
WEAVIATE_PORT = 8080
WEAVIATE_INDEX_NAME = "Vectors"

# Mongo DB
MONGO_HOST = "localhost"
MONGO_PORT = 27017
MONGO_DB_NAME = "Stores"
MONGO_INDEX_STORE_NAMESPACE = "IndexStore"
MONGO_DOC_STORE_NAMESPACE = "DocStore"

# Flags
## TODO: use CLI flags instead
RETRAIN = True
PURGE_DATABASE = True


logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)
logging.getLogger().addHandler(logging.StreamHandler(stream=sys.stdout))


FILE_READER_CLS: Dict[str, Type[BaseReader]] = {
    ".pdf": PDFReaderCustom(image_loader=ImageReaderForFlatPDF(text_type="plain_text")),
    ".jpg": ImageReader,
    ".png": ImageReader,
    ".jpeg": ImageReader,
    ".docx": DocxReader,
}


def main() -> None:
    """Entry point for indexing"""
    
    # llm defintion
    llama_debug = LlamaDebugHandler(print_trace_on_end=True)
    callback_manager = CallbackManager([llama_debug])

    llm = get_llama2(model_path=MODEL_PATH,max_new_tokens=NUM_OUTPUT,model_temperature=0.1,context_window=CONTEXT_WINDOW)
    embed_model = LangchainEmbedding(HuggingFaceEmbeddings(
            model_name=EMBED_MODEL_NAME,
            model_kwargs={"device": "cuda"},
            encode_kwargs={'normalize_embeddings': False}
            ))
    service_context = set_service_ctx(llm=llm, embed_model=embed_model, callback_manager=callback_manager)

    weaviate_client = weaviate.Client(url=f"http://{WEAVIATE_HOST}:{WEAVIATE_PORT}")
    storage_context = set_storage_ctx(weaviate_client)
    storage_indices = load_indices_from_storage(storage_context=storage_context, service_context=service_context)

    if RETRAIN:
        documents = load_data()

        if PURGE_DATABASE:
            purge_vector_store(weaviate_client, WEAVIATE_INDEX_NAME)
        all_doc_ids_memory = [document.doc_id for document in documents]
        all_doc_ids_store = get_doc_ids_in_store(weaviate_client, WEAVIATE_INDEX_NAME)
        doc_ids_to_insert = records_to_insert(all_doc_ids_memory, all_doc_ids_store)
        docs_to_insert = filter_documents_by_doc_ids(documents, doc_ids_to_insert)
        nodes_to_insert=llama_index_preprocessing(docs_to_insert, llm=llm)


        # build vector index
        # It is cheap to index and retrieve the data. We can also reuse the index to answer multiple questions without sending 
        # the documents to LLM many times. The disadvantage is that the quality of the answers depends on the quality of the embeddings.
        # If the embeddings are not good enough, the LLM will not be able to generate a good response. 
        response_synthesizer = get_response_synthesizer(service_context=service_context, use_async=True)
        storage_indices = []
        vector_store_index = VectorStoreIndex(
            nodes=nodes_to_insert,
            storage_context=storage_context,
            service_context=service_context,
            response_synthesizer=response_synthesizer,
            show_progress=True
        )
        storage_indices.extend([vector_store_index])

        # may be a good choice when we have a few questions to answer using a handful of documents.
        # It may give us the best answer because AI will get all the available data, but it is also quite expensive.
        # summary_index = SummaryIndex(
        #     nodes=nodes_to_insert,
        #     storage_context=storage_context,
        #     service_context=service_context,
        #     response_synthesizer=response_synthesizer,
        #     show_progress=True
        # )
        # storage_indices.extend([summary_index])


def custom_completion_to_prompt(completion: str) -> str:
    return completion_to_prompt(
        completion,
        system_prompt=(
            "You are a Q&A assistant. Your goal is to answer questions as "
            "accurately as possible is the instructions and context provided."
        ),
    )

def get_llama2(model_path:str, max_new_tokens:int=256, model_temperature: int=0.1, context_window:int=3800) -> CustomLLM:
    """Init llama-cpp-python https://github.com/abetlen/llama-cpp-python via llama_index.llms"""
    
    # llama2 has a context window of 4096 tokens
    return LlamaCPP(
        model_path=model_path,
        context_window=context_window,
        temperature=model_temperature,
        max_new_tokens=max_new_tokens,
        model_kwargs={"n_gpu_layers": 50, "n_batch": 8, "use_mlock": False},
        messages_to_prompt=messages_to_prompt,
        completion_to_prompt=custom_completion_to_prompt,
        verbose=True)

def set_service_ctx(llm: CustomLLM, embed_model: EmbedType, callback_manager: CallbackManager) -> ServiceContext:
    """Set llamaindex service context"""
    
    service_context = ServiceContext.from_defaults(
        llm=llm,
        chunk_size=CHUNK_SIZE,
        callback_manager=callback_manager,
        embed_model=embed_model,
        prompt_helper=PromptHelper(
            context_window=CONTEXT_WINDOW,
            num_output=NUM_OUTPUT,
            chunk_overlap_ratio=0.2,
            chunk_size_limit=CHUNK_SIZE
        )
    )
    return service_context


def load_weaviate_vector_store(client: weaviate.Client, index: str) -> WeaviateVectorStore:
    """Initiate client connect to weaviate and load llamaindex vector store"""
    
    # embeddings and docs are stored within a Weaviate collection
    return WeaviateVectorStore(weaviate_client=client, index_name=index)

def load_mongo_index_store() -> MongoIndexStore:
    """Load llamaindex mongo index store"""

    return MongoIndexStore.from_host_and_port(
        host=MONGO_HOST,
        port=MONGO_PORT,
        db_name=MONGO_DB_NAME,
        namespace=MONGO_INDEX_STORE_NAMESPACE
    )

def load_mongo_document_store() -> MongoDocumentStore:
    """Load llamaindex's mongo document store"""

    return MongoDocumentStore.from_host_and_port(
        host=MONGO_HOST,
        port=MONGO_PORT,
        db_name=MONGO_DB_NAME,
        namespace=MONGO_DOC_STORE_NAMESPACE
    )

def set_storage_ctx(weaviate_client: weaviate.Client) -> StorageContext:
    """Set llamaindex storage context"""
    
    # load storage context and index
    return StorageContext.from_defaults(
        vector_store=load_weaviate_vector_store(client=weaviate_client, index=WEAVIATE_INDEX_NAME),
        index_store=load_mongo_index_store(),
        docstore=load_mongo_document_store()
    )



def load_data():
    """Custom loading data into llama index documents class"""

    documents = SimpleDirectoryReader(
        input_dir=DATA_PATH,
        filename_as_id=True,
        recursive=True,
        file_extractor=FILE_READER_CLS).load_data()
    return documents


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


def llama_index_preprocessing(documents: List[Document], llm: LLM) -> List[TextNode]:
    """Process (meta) data before indexing"""

    text_splitter=TokenTextSplitter(separator=" ", chunk_size=CHUNK_SIZE, chunk_overlap=20)

    metadata_extractor = MetadataExtractor(
        extractors=[
            #TitleExtractor(nodes=5, llm=llm),
            #QuestionsAnsweredExtractor(questions=2, llm=llm),
            #SummaryExtractor(summaries=["self"], llm=llm),
            #KeywordExtractor(keywords=3, llm=llm),
            EntityExtractor(prediction_threshold=0.5, model_name=DEFAULT_ENTITY_MODEL, device="cuda"),
        ],
        in_place=True,
    )

    node_parser = SimpleNodeParser.from_defaults(
        text_splitter=text_splitter,
        metadata_extractor=metadata_extractor,
        include_metadata=True,
        include_prev_next_rel=True
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
    main()