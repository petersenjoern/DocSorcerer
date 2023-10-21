"""Indexing documents with llamaindex"""

import itertools
import json
import logging
import pickle
import sys
import copy
from pathlib import Path
from typing import Dict, List, Type, Union
from pymongo import MongoClient

import weaviate
from langchain.embeddings import HuggingFaceEmbeddings
from llama_hub.file.image.base import ImageReader as ImageReaderForFlatPDF
from utils.pdf import PDFReaderCustom
from llama_index.prompts import PromptTemplate
from llama_index.schema import IndexNode
from llama_index.embeddings.utils import EmbedType
from llama_index.llms.custom import CustomLLM
from llama_index.llms import HuggingFaceLLM
from llama_index.retrievers import RecursiveRetriever
from llama_index.query_engine import RetrieverQueryEngine

from llama_index import (ServiceContext, SimpleDirectoryReader, VectorStoreIndex, get_response_synthesizer)
from llama_index.callbacks import (CallbackManager,LlamaDebugHandler)
from llama_index.embeddings.langchain import LangchainEmbedding
from llama_index.indices.loading import load_indices_from_storage
from llama_index.indices.prompt_helper import PromptHelper
from llama_index.llms import LlamaCPP
from llama_index.llms.llama_utils import (completion_to_prompt,
                                          messages_to_prompt)
from llama_index.node_parser import SimpleNodeParser
from llama_index.node_parser.extractors import (MetadataExtractor,
                                                QuestionsAnsweredExtractor,
                                                SummaryExtractor)

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
CONTEXT_WINDOW = 3500
NUM_OUTPUT = 596
CHUNK_SIZE = 1024

# Embedding Model
EMBED_MODEL_NAME = "BAAI/bge-small-en"
EMBED_MODEL = LangchainEmbedding(
    HuggingFaceEmbeddings(
        model_name=EMBED_MODEL_NAME,
        model_kwargs={"device": "cuda"},
        encode_kwargs={'normalize_embeddings': False}
    )
)

# LlamaIndex Callback
CALLBACK_MANAGER = CallbackManager([LlamaDebugHandler(print_trace_on_end=True)])


# Data
DATA_PATH = Path.cwd().joinpath("data")
DATA_SOURCE_PATH = DATA_PATH.joinpath("source", "Leisure")

# Index Node Ref
DATA_METADATA_PATH = DATA_PATH.joinpath("indexing", "metadata_dicts.jsonl")
NODE_REFERENCES_PATH = DATA_PATH.joinpath("indexing", "node_references.pickle")

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
PURGE_DATABASES = False


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

    llm = get_llama2(
        model_path=MODEL_PATH,
        max_new_tokens=NUM_OUTPUT,
        model_temperature=0.1,
        context_window=CONTEXT_WINDOW
    )
    #llm = get_huggingface_llm(
    #    model_name="Writer/camel-5b-hf",
    #    max_new_tokens=NUM_OUTPUT,
    #    model_temperature=0.1,
    #    context_window=CONTEXT_WINDOW
    #)

    weaviate_client = weaviate.Client(url=f"http://{WEAVIATE_HOST}:{WEAVIATE_PORT}")
    mongodb_client = MongoClient(host=MONGO_HOST, port=MONGO_PORT)
    
    service_context = set_service_ctx(llm=llm, embed_model=EMBED_MODEL, callback_manager=CALLBACK_MANAGER)
    storage_context = set_storage_ctx(weaviate_client)
    storage_indices = load_indices_from_storage(storage_context=storage_context, service_context=service_context)

        
    if PURGE_DATABASES:
        purge_weaviate_schema(weaviate_client, WEAVIATE_INDEX_NAME)
        purge_mongo_database(mongodb_client, MONGO_DB_NAME)
        purge_node_references(path=NODE_REFERENCES_PATH)
     
    documents = load_data_from_path(input_dir=DATA_SOURCE_PATH, collect_pages=True)

    # Filter out documents that already have been indexed; this is based on the document filesystem path location
    # Meaning, it is not recognised if the document on the filesystem has been modified after last ingestion
    all_doc_ids_memory = [document.doc_id for document in documents]
    all_doc_ids_store = get_doc_ids_in_store(weaviate_client, WEAVIATE_INDEX_NAME)
    doc_ids_to_insert = records_to_insert(all_doc_ids_memory, all_doc_ids_store)
    docs_to_insert = filter_documents_by_doc_ids(documents, doc_ids_to_insert)


    base_node_parser = SimpleNodeParser.from_defaults(
        text_splitter=TokenTextSplitter(separator=" ", chunk_size=CHUNK_SIZE, chunk_overlap=20),
        callback_manager=CALLBACK_MANAGER
    )

    # this will auto-generate a new TextNode(id_="") UUID unfortunately
    # meaning that even though the document hasnt changed, the UUID will be another one
    # this means that the metadata from the MetadataExtractor id_ will not match the newly generated TextNode(id_="")  
    base_nodes = base_node_parser.get_nodes_from_documents(documents=docs_to_insert, show_progress=True)

    metadata_extractor = MetadataExtractor(
        extractors=[
            SummaryExtractor(llm=llm, summaries=["self"], show_progress=True),
            QuestionsAnsweredExtractor(llm=llm, questions=5, show_progress=True),
        ],
    )


    base_nodes_metadata_dicts = []
    # we always want to append new metadata (never delete!), as we will match the metadata with the UUID ("id_" field)
    try:
        base_nodes_metadata_dicts=load_metadata_dicts(path=DATA_METADATA_PATH)
    except FileNotFoundError:
        logging.warning("You may have screwed up your references. ",
                        "If this is the first time running the indexing you may be good, ",
                        "Check if you have already a base_nodes_metadata_dicts in your DATA_METADATA_PATH.")
        # raise Exception("There is no metadata for the base nodes (TextNode). Check the Path.")

    for base_node in base_nodes:
        # extractor.extract(nodes) is only return the extraction result, but loosing its metadata
        # this loop is preserving metadata that is required later on (matching on node-ids, required to be UUID)
        metadata_dicts = metadata_extractor.extract([base_node])
        metadata_dicts[0]["id_"] = base_node.id_
        metadata_dicts[0]["ref_doc_id"] = base_node.ref_doc_id
        base_nodes_metadata_dicts.extend(metadata_dicts)

    save_metadata_dicts(path=DATA_METADATA_PATH, dicts=base_nodes_metadata_dicts)

    # base_nodes_already_indexed = get_text_nodes_from_store(weaviate_client, WEAVIATE_INDEX_NAME)
    # base_nodes.extend(base_nodes_already_indexed)
    
    
    # all_nodes will consist eventually of all base_nodes (TextNode), linked to its metadata (IndexNode)
    # loading all previous indexed TextNode and IndexNode into.
    # The assumption is that these TextNode and IndexNode are still in the database, and we are appending to it below.
    # This is for simplicity right now
    # TODO: get actual TextNode and IndexNode from the database(s)
    indexed_nodes =[]
    if not PURGE_DATABASES:
        try:
            indexed_nodes = load_node_references(NODE_REFERENCES_PATH)
        except FileNotFoundError:
            logging.warning("You have screwed up and lost your references. ",
                            "The retrieval of references from the DB is not supported yet. ",
                            "Delete your base_nodes_metadata_dicts file and start over again.")
    indexed_nodes_ids = [n.id_ for n in indexed_nodes]
    nodes_to_be_indexed = copy.deepcopy(base_nodes)
    nodes_to_be_indexed_ids = [n.id_ for n in nodes_to_be_indexed]
    for metadata_dict in base_nodes_metadata_dicts:
        # only add metadata to the Index for documents/nodes that are in memory
        if (metadata_dict["id_"] not in indexed_nodes_ids) and (metadata_dict["id_"] in nodes_to_be_indexed_ids):
            inode_q = IndexNode(
                text=metadata_dict["questions_this_excerpt_can_answer"],
                index_id=metadata_dict["id_"],
                node_id=f'{metadata_dict["id_"]}-questions'
            )
            inode_s = IndexNode(
                text=metadata_dict["section_summary"],
                index_id=metadata_dict["id_"],
                node_id=f'{metadata_dict["id_"]}-summary'
            )
            nodes_to_be_indexed.extend([inode_q, inode_s])
    
   

    # build vector index
    # It is cheap to index and retrieve the data. We can also reuse the index to answer multiple questions without sending 
    # the documents to LLM many times. The disadvantage is that the quality of the answers depends on the quality of the embeddings.
    # If the embeddings are not good enough, the LLM will not be able to generate a good response. 
    response_synthesizer = get_response_synthesizer(service_context=service_context, use_async=True)
    storage_indices = []
    vector_store_index = VectorStoreIndex(
        nodes=nodes_to_be_indexed,
        storage_context=storage_context,
        service_context=service_context,
        response_synthesizer=response_synthesizer,
        show_progress=True
    )
    storage_indices.extend([vector_store_index])

    all_nodes: List[Union[IndexNode, TextNode]] = indexed_nodes + nodes_to_be_indexed
    save_node_references(NODE_REFERENCES_PATH, all_nodes)


def save_node_references(path: Path, _dict: Dict[str, Union[TextNode, IndexNode]]) -> None:
    """Save dictionary with reference objects under path"""
    with open(path, 'wb') as handle:
        pickle.dump(_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)

def load_node_references(path: Path) -> Dict[str, Union[TextNode, IndexNode]]:
    """Load dictionary with reference objects from path"""
    with open(path, 'rb') as handle:
        return pickle.load(handle)


def save_metadata_dicts(path: Path, dicts: List[Dict[str, str]]) -> None:
    """Save list of dictionaries under path"""
    with open(path, "w") as fp:
        for m in dicts:
            fp.write(json.dumps(m) + "\n")


def load_metadata_dicts(path: str) -> List[Dict[str, str]]:
    """Load list of dictionaries from path"""
    with open(path, "r") as fp:
        json_list = list(fp)
        metadata_dicts = [json.loads(json_string) for json_string in json_list]
        return metadata_dicts

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

def get_huggingface_llm(model_name:str, max_new_tokens:int=256, model_temperature: int=0.1, context_window:int=2048) -> HuggingFaceLLM:
    """Return a hugginface LLM"""

    query_wrapper_prompt = PromptTemplate(
        "Below is an instruction that describes a task. "
        "Write a response that appropriately completes the request.\n\n"
        "### Instruction:\n{query_str}\n\n### Response:"
    )

    return HuggingFaceLLM(
        model_name=model_name,
        tokenizer_name=model_name,
        context_window=context_window,
        max_new_tokens=max_new_tokens,
        generate_kwargs={"temperature": model_temperature, "do_sample": True},
        device_map="auto",
        tokenizer_kwargs={"max_length": 2048},
        query_wrapper_prompt=query_wrapper_prompt,
        model_kwargs={"max_memory": {0: "18GB"}, "offload_folder": "/tmp/offload"}
    )


def set_service_ctx(
        llm: Union[CustomLLM, HuggingFaceLLM],
        embed_model: EmbedType, callback_manager: CallbackManager) -> ServiceContext:
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

def load_data_from_path(input_dir: Path, collect_pages: bool=True) -> List[Document]:
    """Custom loading data into llama index documents class"""

    documents_to_return = []
    loaded_documents = SimpleDirectoryReader(
        input_dir=input_dir,
        filename_as_id=True,
        recursive=True,
        file_extractor=FILE_READER_CLS).load_data()
    documents_to_return = loaded_documents
    
    if collect_pages:
        collecting_pages_func = lambda doc:Path(doc.node_id).with_suffix("")
        documents_grouped_by_title_iter = itertools.groupby(loaded_documents, collecting_pages_func)
        
        documents_collected = []
        for _, documents_for_title_iter in documents_grouped_by_title_iter:
            documents_for_title = list(documents_for_title_iter)
            try:
                # for pdf files, the metadata/extrainfo property contains the file_name
                # this needs to be joined with parts of the doc_id
                file_name = documents_for_title[0].metadata["file_name"]
                full_file_name = Path(documents_for_title[0].doc_id).parent.joinpath(file_name)
            except:
                # for other file formats, the doc_id has the full correct path to the file
                full_file_name = Path(documents_for_title[0].doc_id)
            doc_text = "\n\n".join([d.get_content() for d in documents_for_title])
            docs = Document(text=doc_text, doc_id=str(full_file_name))
            documents_collected.append(docs)
        documents_to_return = documents_collected

    return documents_to_return


def records_to_insert(inmemory_records: List[str], stored_records: List[str]) -> List[str]:
    """Records not stored, to be inserted."""
    return list(set(inmemory_records) - set(stored_records))

def records_to_update(stored_records: List[str], inmemory_records: List[str]) -> List[str]:
    """potential records to update."""
    return list(set(stored_records).intersection(inmemory_records))


def get_doc_ids_in_store(client: weaviate.Client, class_name: str) -> List[str]:
    """Return all doc ids in weaviate store"""

    try:
        client.schema.get(class_name)
    except weaviate.UnexpectedStatusCodeException:
        return []

    all_objects_store = client.data_object.get(class_name=class_name)
    all_doc_ids_store = [i["properties"]["doc_id"] for i in all_objects_store["objects"]]
    return all_doc_ids_store

def get_node_ids_in_store(client: weaviate.Client, class_name: str) -> List[str]:
    """Return the node.id (also IndexNode/TextNode("id_") field) from weaviate store"""

    try:
        client.schema.get(class_name)
    except weaviate.UnexpectedStatusCodeException:
        return []
    
    all_objects_store = client.data_object.get(class_name=class_name)
    all_node_ids_store = [i["properties"]["id"] for i in all_objects_store["objects"]]
    return all_node_ids_store

def get_text_nodes_from_store(client: weaviate.Client, class_name: str) -> List[TextNode]:
    """Return all TextNodes from weaviate store"""

    class_properties = ["doc_id", "document_id", "ref_doc_id", "text"]
    batch_size = 50
    cursor=None
    query = (
        client.query.get(class_name, class_properties)
        # Optionally retrieve the vector embedding by adding `vector` to the _additional fields
        #.with_additional(["id vector"])
        .with_limit(batch_size)
    )

    if cursor is not None:
        res = query.with_after(cursor).do()
    else:
        res = query.do()
    
    return res

def purge_weaviate_schema(client: weaviate.Client, class_name: str):
    """Purge vector store for a schema (class name)"""
    try:
        client.schema.delete_class(class_name)
    except:
        pass

def purge_mongo_database(client: MongoClient, database_name: str) -> None:
    """Purge database within MongoDB"""

    client.drop_database(database_name)

def purge_node_references(path: Path) -> None:
    """Delete the pickle file on the path"""

    if path.exists() and path.suffix == ".pickle":
        path.unlink()


def filter_documents_by_doc_ids(documents: List[Document], doc_ids: List[str]) -> List[Document]:
    """Filter documents by doc ids"""

    result = filter(lambda x: x.doc_id in doc_ids, documents)
    return list(result)


if __name__ == "__main__":
    main()