"""Module related to handling storage used in indexing/API."""

from pymongo import MongoClient
import weaviate
from llama_index.storage.storage_context import StorageContext
from llama_index.storage.docstore import MongoDocumentStore
from llama_index.storage.index_store import MongoIndexStore
from llama_index.vector_stores.weaviate import WeaviateVectorStore

# Weaviate
WEAVIATE_INDEX_NAME = "Vectors"

# Mongo DB
MONGO_HOST = "localhost"
MONGO_PORT = 27017
MONGO_DB_NAME = "Stores"
MONGO_INDEX_STORE_NAMESPACE = "IndexStore"
MONGO_DOC_STORE_NAMESPACE = "DocStore"


def set_storage_ctx(weaviate_client: weaviate.Client) -> StorageContext:
    """Set llamaindex storage context"""
    
    # load storage context and index
    return StorageContext.from_defaults(
        vector_store=load_weaviate_vector_store(client=weaviate_client),
        index_store=load_mongo_index_store(),
        docstore=load_mongo_document_store()
    )


def load_weaviate_vector_store(client: weaviate.Client) -> WeaviateVectorStore:
    """Initiate client connect to weaviate and load llamaindex vector store"""
    
    # embeddings and docs are stored within a Weaviate collection
    return WeaviateVectorStore(weaviate_client=client, index_name=WEAVIATE_INDEX_NAME)


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


def purge_all_indices(weaviate_client, mongodb_client):
    """Purge all indices and document references."""

    purge_weaviate_schema(weaviate_client, WEAVIATE_INDEX_NAME)
    purge_mongo_database(mongodb_client, MONGO_DB_NAME)


def purge_weaviate_schema(client: weaviate.Client, class_name: str):
    """Purge vector store for a schema (class name)"""

    try:
        client.schema.delete_class(class_name)
    except:
        pass


def purge_mongo_database(client: MongoClient, database_name: str) -> None:
    """Purge database within MongoDB"""

    client.drop_database(database_name)