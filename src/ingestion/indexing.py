"""Indexing documents with llamaindex"""

import copy
import itertools
import json
import logging
import pickle
from pathlib import Path
from typing import Dict, List, Type, Union

import weaviate
from langchain.embeddings import HuggingFaceEmbeddings
from llama_hub.file.image.base import ImageReader as ImageReaderForFlatPDF
from llama_index import (
    ServiceContext,
    SimpleDirectoryReader,
    VectorStoreIndex,
    get_response_synthesizer,
)
from llama_index.callbacks import CallbackManager, LlamaDebugHandler
from llama_index.embeddings.langchain import LangchainEmbedding
from llama_index.indices.loading import load_indices_from_storage
from llama_index.indices.prompt_helper import PromptHelper
from llama_index.node_parser import SimpleNodeParser
from llama_index.node_parser.extractors import (
    MetadataExtractor,
    QuestionsAnsweredExtractor,
    SummaryExtractor,
)
from llama_index.readers.base import BaseReader
from llama_index.readers.file.docs_reader import DocxReader
from llama_index.readers.file.image_reader import ImageReader
from llama_index.schema import Document, IndexNode, TextNode
from llama_index.text_splitter.token_splitter import TokenTextSplitter
from pymongo import MongoClient
from utils.pdf import PDFReaderCustom

from config import Settings
from models.language_models import get_llama2
from storage.llamaindex_storage import purge_dbs, set_storage_ctx

settings = Settings()


# Embedding Model
EMBED_MODEL = LangchainEmbedding(
    HuggingFaceEmbeddings(
        model_name=settings.embed.name,
        model_kwargs={"device": "cuda"},
        encode_kwargs={"normalize_embeddings": False},
    )
)
# Prompt Helper - can help deal with LLM context window token limitations
PROMPT_HELPER_INDEXING = PromptHelper(
    context_window=settings.prompt_helper.context_window,
    num_output=settings.prompt_helper.num_output,
    chunk_overlap_ratio=settings.prompt_helper.chunk_overlap_ratio,
    chunk_size_limit=settings.parser.chunk_size,
)

# LlamaIndex Callback
LLAMA_INDEX_CALLBACKS = CallbackManager([LlamaDebugHandler(print_trace_on_end=True)])


BASE_NODE_PARSER = SimpleNodeParser.from_defaults(
    text_splitter=TokenTextSplitter(
        separator=settings.parser.separator,
        chunk_size=settings.parser.chunk_size,
        chunk_overlap=settings.parser.chunk_overlap,
    ),
    callback_manager=LLAMA_INDEX_CALLBACKS,
)

# Data
DATA_PATH = Path.cwd().joinpath("data")
DATA_SOURCE_PATH = DATA_PATH.joinpath("source", "Leisure")

# Index Node Ref
DATA_METADATA_PATH = DATA_PATH.joinpath("indexing", "metadata_dicts.jsonl")
NODE_REFERENCES_PATH = DATA_PATH.joinpath("indexing", "node_references.pickle")

# Flags
## TODO: use CLI flags instead
PURGE_ALL = False


logging.basicConfig(
    filename="indexing.log",
    format="%(asctime)s - %(message)s",
    level=logging.DEBUG,
)
logger = logging.getLogger(__name__)


FILE_READER_CLS: Dict[str, Type[BaseReader]] = {
    ".pdf": PDFReaderCustom(image_loader=ImageReaderForFlatPDF(text_type="plain_text")),
    ".jpg": ImageReader,
    ".png": ImageReader,
    ".jpeg": ImageReader,
    ".docx": DocxReader,
}


def main() -> None:
    """Entry point for indexing"""

    llm = get_llama2(
        max_new_tokens=settings.llm.context_window,
        model_temperature=settings.llm.temperature,
        context_window=settings.llm.num_output,
    )

    weaviate_client = weaviate.Client(
        url=f"http://{settings.db_vector.host}:{settings.db_vector.port}"
    )
    mongodb_client = MongoClient(
        host=settings.db_no_sql.host, port=settings.db_no_sql.port
    )

    service_context = ServiceContext.from_defaults(
        llm=llm,
        chunk_size=settings.parser.chunk_size,
        callback_manager=LLAMA_INDEX_CALLBACKS,
        embed_model=EMBED_MODEL,
        prompt_helper=PROMPT_HELPER_INDEXING,
    )
    storage_context = set_storage_ctx(weaviate_client)
    storage_indices = load_indices_from_storage(storage_context=storage_context)

    if PURGE_ALL:
        purge_dbs(weaviate_client, mongodb_client)
        purge_node_references(path=NODE_REFERENCES_PATH)

    documents = load_data_from_path(input_dir=DATA_SOURCE_PATH, collect_pages=True)

    # Filter out documents that already have been indexed; this is based on the document filesystem path location
    # Meaning, it is not recognised if the document on the filesystem has been modified after last ingestion
    docs_to_insert = documents_to_insert(weaviate_client, documents)

    # document will be parsed into Node(s) with auto-generated TextNode(id_="") UUID
    # meaning that even though the document hasnt changed, the UUID for each Node(s) will be never the same
    # this means that the metadata from the MetadataExtractor id_ will not match the newly generated TextNode(id_="")
    base_nodes = BASE_NODE_PARSER.get_nodes_from_documents(
        documents=docs_to_insert, show_progress=True
    )

    metadata_extractor = MetadataExtractor(
        extractors=[
            SummaryExtractor(llm=llm, summaries=["self"], show_progress=True),
            QuestionsAnsweredExtractor(llm=llm, questions=5, show_progress=True),
        ],
    )

    base_nodes_metadata_dicts = generate_metadata_from_base_nodes(
        base_nodes, metadata_extractor
    )

    # "all_nodes" will consist eventually of all base_nodes (TextNode), linked to its metadata (IndexNode)
    # The assumption is that TextNode and IndexNode are still in the database and mirror the objects in the
    # object under "NODE_REFERENCES_PATH". We will append new Text- and IndexNode(s) to this object.
    # TODO: get actual TextNode and IndexNode from the database(s) instead of  the object under the "NODE_REFERENCES_PATH"
    indexed_nodes, nodes_to_be_indexed = separate_nodes_by_index_status(
        base_nodes, base_nodes_metadata_dicts
    )

    # build vector index
    # It is cheap to index and retrieve the data. We can also reuse the index to answer multiple questions without sending
    # the documents to LLM many times. The disadvantage is that the quality of the answers depends on the quality of the embeddings.
    # If the embeddings are not good enough, the LLM will not be able to generate a good response.
    response_synthesizer = get_response_synthesizer(
        service_context=service_context, use_async=True
    )
    storage_indices = []
    vector_store_index = VectorStoreIndex(
        nodes=nodes_to_be_indexed,
        storage_context=storage_context,
        service_context=service_context,
        response_synthesizer=response_synthesizer,
        show_progress=True,
    )
    storage_indices.extend([vector_store_index])

    # Finally, after indexing the new nodes (TextNode and IndexNode), we want to refresh the object will all node references
    # so that the object is reflecting the database status. This is a workaround for now.
    all_nodes: List[Union[IndexNode, TextNode]] = indexed_nodes + nodes_to_be_indexed
    save_node_references(NODE_REFERENCES_PATH, all_nodes)


def purge_node_references(path: Path) -> None:
    """Delete the pickle file on the path"""

    if path.is_file() and path.suffix == ".pickle":
        path.unlink()


def load_data_from_path(input_dir: Path, collect_pages: bool = True) -> List[Document]:
    """Custom loading data into llama index documents class"""

    documents_to_return = []
    loaded_documents = SimpleDirectoryReader(
        input_dir=input_dir,
        filename_as_id=True,
        recursive=True,
        file_extractor=FILE_READER_CLS,
    ).load_data()
    documents_to_return = loaded_documents

    def collecting_pages_func(doc):
        return Path(doc.node_id).with_suffix("")

    if collect_pages:
        documents_grouped_by_title_iter = itertools.groupby(
            loaded_documents, collecting_pages_func
        )

        documents_collected = []
        for _, documents_for_title_iter in documents_grouped_by_title_iter:
            documents_for_title = list(documents_for_title_iter)
            try:
                # for pdf files, the metadata/extrainfo property contains the file_name
                # this needs to be joined with parts of the doc_id
                file_name = documents_for_title[0].metadata["file_name"]
                full_file_name = Path(documents_for_title[0].doc_id).parent.joinpath(
                    file_name
                )
            except:
                # for other file formats, the doc_id has the full correct path to the file
                full_file_name = Path(documents_for_title[0].doc_id)
            doc_text = "\n\n".join([d.get_content() for d in documents_for_title])
            docs = Document(text=doc_text, doc_id=str(full_file_name))
            documents_collected.append(docs)
        documents_to_return = documents_collected

    return documents_to_return


def documents_to_insert(
    weaviate_client: weaviate.Client, documents: List[Document]
) -> List[Document]:
    """Based on doc_ids in memory and doc_ids in index, only return documents to be inserted."""

    all_doc_ids_memory = [document.doc_id for document in documents]
    all_doc_ids_store = _get_doc_ids_in_store(
        weaviate_client, settings.db_vector.collection.name
    )
    doc_ids_to_insert = _records_to_insert(all_doc_ids_memory, all_doc_ids_store)
    docs_to_insert = _filter_documents_by_doc_ids(documents, doc_ids_to_insert)
    return docs_to_insert


def _get_doc_ids_in_store(client: weaviate.Client, class_name: str) -> List[str]:
    """Return all doc ids in weaviate store"""

    try:
        client.schema.get(class_name)
    except weaviate.UnexpectedStatusCodeException:
        return []

    # TODO: make this smarter, to only return ids for unlimited amount of docs
    all_objects_store = client.data_object.get(class_name=class_name, limit=10_000)
    all_doc_ids_store = [
        i["properties"]["doc_id"] for i in all_objects_store["objects"]
    ]
    return all_doc_ids_store


def _records_to_insert(
    inmemory_records: List[str], stored_records: List[str]
) -> List[str]:
    """Records not stored, to be inserted."""

    return list(set(inmemory_records) - set(stored_records))


def _filter_documents_by_doc_ids(
    documents: List[Document], doc_ids: List[str]
) -> List[Document]:
    """Filter documents by doc ids"""

    result = filter(lambda x: x.doc_id in doc_ids, documents)
    return list(result)


def generate_metadata_from_base_nodes(
    base_nodes: List[TextNode], metadata_extractor: MetadataExtractor
) -> List[Dict[str, str]]:
    """Generate metadata (with help of an LLM) from base_nodes and append to a metadata dictionary."""

    base_nodes_metadata_dicts = []
    # we always want to append new metadata (never delete!), as we will match the metadata with the UUID ("id_" field)
    try:
        base_nodes_metadata_dicts = _load_metadata_dicts(path=DATA_METADATA_PATH)
    except FileNotFoundError:
        logger.warning(
            "You may have screwed up your references. ",
            "If this is the first time running the indexing you may be good, ",
            "Check if you have already a base_nodes_metadata_dicts in your DATA_METADATA_PATH.",
        )

    base_nodes_metadata_dicts = _extract_metadata_from_nodes(
        base_nodes, metadata_extractor, base_nodes_metadata_dicts
    )
    _save_metadata_dicts(path=DATA_METADATA_PATH, dicts=base_nodes_metadata_dicts)
    return base_nodes_metadata_dicts


def _load_metadata_dicts(path: str) -> List[Dict[str, str]]:
    """Load list of dictionaries from path"""

    with open(path, "r") as fp:
        json_list = list(fp)
        metadata_dicts = [json.loads(json_string) for json_string in json_list]
        return metadata_dicts


def _extract_metadata_from_nodes(
    base_nodes: List[TextNode],
    metadata_extractor: MetadataExtractor,
    base_nodes_metadata_dicts: List[Dict[str, str]],
) -> List[Dict[str, str]]:
    """For every base_node extract metadata and append additional information, such as the id_ from the base_node to it."""

    for base_node in base_nodes:
        # extractor.extract(nodes) is only return the extraction result, but loosing its metadata
        # this loop is preserving metadata that is required later on (matching on node["id_"], required to be UUID)
        metadata_dicts = metadata_extractor.extract([base_node])
        metadata_dicts[0]["id_"] = base_node.id_
        metadata_dicts[0]["ref_doc_id"] = base_node.ref_doc_id
        base_nodes_metadata_dicts.extend(metadata_dicts)
    return base_nodes_metadata_dicts


def _save_metadata_dicts(path: Path, dicts: List[Dict[str, str]]) -> None:
    """Save list of dictionaries under path"""

    with open(path, "w") as fp:
        for m in dicts:
            fp.write(json.dumps(m) + "\n")


def separate_nodes_by_index_status(
    base_nodes: List[TextNode], base_nodes_metadata_dicts: List[Dict[str, str]]
):
    """By loading the existing node references, calculate which (new) nodes have to be indexed."""

    indexed_nodes: List[Union[IndexNode, TextNode]] = []
    if not PURGE_ALL:
        try:
            indexed_nodes = _load_node_references(NODE_REFERENCES_PATH)
        except FileNotFoundError:
            logger.warning(
                "You have screwed up and lost your references. ",
                "The retrieval of references from the DB is not supported yet. ",
                "Delete your base_nodes_metadata_dicts file and start over again.",
            )
    indexed_nodes_ids = [n.id_ for n in indexed_nodes]
    nodes_to_be_indexed = _prepare_nodes_to_be_indexed(
        base_nodes, base_nodes_metadata_dicts, indexed_nodes_ids
    )
    return indexed_nodes, nodes_to_be_indexed


def _load_node_references(path: Path) -> List[Union[TextNode, IndexNode]]:
    """Load dictionary with reference objects from path"""

    with open(path, "rb") as handle:
        return pickle.load(handle)


def _prepare_nodes_to_be_indexed(
    base_nodes: List[TextNode],
    base_nodes_metadata_dicts: List[Dict[str, str]],
    indexed_nodes_ids: List[str],
) -> List[Union[TextNode, IndexNode]]:
    """Add metadata (IndexNode) to all nodes that still need to be indexed.
    Avoid adding duplicated metadata by checking the indexed node ids"""
    nodes_to_be_indexed = copy.deepcopy(base_nodes)
    nodes_to_be_indexed_ids = [n.id_ for n in nodes_to_be_indexed]
    for metadata_dict in base_nodes_metadata_dicts:
        # only add metadata to the Index for documents/nodes that are in memory
        if (metadata_dict["id_"] not in indexed_nodes_ids) and (
            metadata_dict["id_"] in nodes_to_be_indexed_ids
        ):
            inode_q = IndexNode(
                text=metadata_dict["questions_this_excerpt_can_answer"],
                index_id=metadata_dict["id_"],
            )
            inode_s = IndexNode(
                text=metadata_dict["section_summary"], index_id=metadata_dict["id_"]
            )
            nodes_to_be_indexed.extend([inode_q, inode_s])
    return nodes_to_be_indexed


def save_node_references(path: Path, nodes: List[Union[TextNode, IndexNode]]) -> None:
    """Save dictionary with reference objects under path"""

    with open(path, "wb") as handle:
        pickle.dump(nodes, handle, protocol=pickle.HIGHEST_PROTOCOL)


if __name__ == "__main__":
    main()
