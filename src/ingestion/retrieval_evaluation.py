"""Evaluate the retrieval and response of the system"""

import asyncio
import itertools
import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Awaitable, Dict, Generator, List, Optional, Tuple, Union

import pandas as pd
import pymongo
import weaviate
from config import Settings
from indexing import load_data_from_path
from langchain.embeddings import HuggingFaceEmbeddings
from language_models import get_zephyr
from llama_index import (
    LangchainEmbedding,
    ServiceContext,
    StorageContext,
    SummaryIndex,
    VectorStoreIndex,
)
from llama_index.evaluation import RetrieverEvaluator, generate_question_context_pairs
from llama_index.evaluation.retrieval.base import RetrievalEvalResult
from llama_index.finetuning.embeddings.common import EmbeddingQAFinetuneDataset
from llama_index.indices.vector_store.retrievers import VectorIndexAutoRetriever
from llama_index.llms.base import LLM
from llama_index.node_parser import SimpleNodeParser
from llama_index.retrievers import QueryFusionRetriever, VectorIndexRetriever
from llama_index.retrievers.fusion_retriever import FUSION_MODES
from llama_index.schema import TextNode
from llama_index.storage.docstore import MongoDocumentStore
from llama_index.storage.index_store import MongoIndexStore
from llama_index.text_splitter.token_splitter import TokenTextSplitter
from llama_index.vector_stores.types import (
    MetadataInfo,
    VectorStoreInfo,
    VectorStoreQueryMode,
)
from llama_index.vector_stores.weaviate import WeaviateVectorStore
from llamaindex_storage import purge_mongo_database, purge_weaviate_schema

logging.basicConfig(
    filename="retrieval_evaluation.log",
    format="%(asctime)s - %(message)s",
    level=logging.DEBUG,
)
logger = logging.getLogger(__name__)

settings = Settings()
settings.db_vector.collection.name = "RetrievalEval"
settings.db_no_sql.collection_document.name = "RetrievalEvalDoc"
settings.db_no_sql.collection_index.name = "RetrievalEvalIndex"
settings.db_no_sql.database.name = "RetrievalEval"

DATA_PATH = Path.cwd().joinpath("data")
EVAL_DOCS_PATH = DATA_PATH.joinpath("source", "evaluation", "retrieval", "documents")
EVAL_DOCS_TO_FIND = EVAL_DOCS_PATH.joinpath("to_find")
EVAL_DOCS_ADD_NOISE = EVAL_DOCS_PATH.joinpath("add_noise")
EVAL_RETRIEVAL_NODES_TO_RETRIEVE = DATA_PATH.joinpath(
    "cache", "evaluation", "retrieval", "nodes", "to_find"
)
EVAL_RETRIEVAL_NODES_ADD_NOISE = DATA_PATH.joinpath(
    "cache", "evaluation", "retrieval", "nodes", "add_noise"
)
EVAL_RETRIEVAL_NODES_QA_GENERATED = DATA_PATH.joinpath(
    "cache", "evaluation", "retrieval", "qa", "generated"
)
EVAL_RETRIEVAL_NODES_QA_APPROVED = DATA_PATH.joinpath(
    "cache", "evaluation", "retrieval", "qa", "approved"
)


TITLE_METADATA_SPLIT_PATTERN = "__m__"
BASE_NODES_FNAME_PATTERN = "{title}{split_pattern}{chunk_size}_{chunk_overlap}.json"


@dataclass
class NodesForTitle:
    title: str
    nodes: List[TextNode]
    chunk_size: int
    chunk_overlap: int


metadata_memory = {
    "Deep-Work-JP": {
        "title": "Deep Work",
        "category_sub_sub": "Productivity",
        "author": "Cal Newport",
        "published": "2018",
    },
    "Barking-up-the-wrong-tree-JP": {
        "title": "Barking up the wrong tree",
        "category_sub_sub": "Self-help",
        "author": "Eric Barker",
        "published": "2017",
    },
}

VECTOR_STORE_QUERY_MODES = [
    # VectorStoreQueryMode.DEFAULT,
    VectorStoreQueryMode.HYBRID,
]
VECTOR_STORE_QUERY_HYBRID_ALPHA = [
    0.75,
    0.6,
    0.5,
]  # alpha = 0 -> bm25, alpha=1 -> vector search, only working with hybrid
SIMILARITY_TOP_K = [2]
QUERY_FUSION_RETRIEVER_MODES = [
    FUSION_MODES.SIMPLE,
    # FUSION_MODES.RECIPROCAL_RANK
]
QUERY_FUSION_RETRIEVER_NUM_QUERIES = [2]  # set this to 1 to disable query generation

HYPERPARAMETERS_VECTOR_INDEX_RETRIEVER = {
    "similarity_top_k": SIMILARITY_TOP_K,
    "vector_store_query_mode": VECTOR_STORE_QUERY_MODES,
    "alpha": VECTOR_STORE_QUERY_HYBRID_ALPHA,
}

HYPERPARAMETERS_FUSION_RETRIEVER = {
    "similarity_top_k": SIMILARITY_TOP_K,
    "vector_store_query_mode": VECTOR_STORE_QUERY_MODES,
    "alpha": VECTOR_STORE_QUERY_HYBRID_ALPHA,
    "num_queries": QUERY_FUSION_RETRIEVER_NUM_QUERIES,
    "mode": QUERY_FUSION_RETRIEVER_MODES,
}

HYPERPARAMETERS_VECTOR_INDEX_AUTO_RETRIEVER = {
    "similarity_top_k": SIMILARITY_TOP_K,
    "vector_store_query_mode": VECTOR_STORE_QUERY_MODES,
    "alpha": VECTOR_STORE_QUERY_HYBRID_ALPHA,
}


# Define a list embedding models.
EMBEDDING_MODELS = ["BAAI/bge-small-en-v1.5", "BAAI/llm-embedder"]


# TODO: try prompts
# TODO: try deep memory


async def retrieval_evaluation_and_report():
    """
    Create and evaluate multiple retrievers, then report their results.
    """
    nodes_to_retrieve, nodes_add_noise = prepare_nodes()

    llm = prepare_llm()

    generate_and_save_qa_dataset(
        titles_with_nodes=nodes_to_retrieve,
        path=EVAL_RETRIEVAL_NODES_QA_GENERATED,
        llm=llm,
    )

    langchain_embeddings = create_embeddings()

    retrieval_results = await retrieve_and_evaluate(
        nodes_to_retrieve,
        nodes_add_noise,
        langchain_embeddings,
        llm,
    )

    report_retrieval_results(retrieval_results)


def report_retrieval_results(data):
    # Convert list of dicts to pandas DataFrame
    df = pd.DataFrame(data)
    print(df)


def prepare_nodes() -> Tuple[List[NodesForTitle], List[NodesForTitle]]:
    """
    Prepares and retrieves two lists of nodes: one for retrieval and one to add noise.

    This function first ensures all nodes are created and saved for the current documents.
    Then, it loads two sets of nodes:
    - Nodes to be retrieved which are the main focus of the retrieval process.
    - Nodes to add noise which are used to make the retrieval scenario more realistic.

    Returns:
        A tuple of two lists:
        - The first list contains nodes that should be retrieved.
        - The second list contains nodes that are used to add noise.
    """
    create_and_save_nodes_for_all_docs()
    nodes_to_retrieve = load_nodes_with_title_to_retrieve(
        EVAL_RETRIEVAL_NODES_TO_RETRIEVE
    )
    nodes_add_noise = load_nodes_with_title_to_retrieve(EVAL_RETRIEVAL_NODES_ADD_NOISE)
    return nodes_to_retrieve, nodes_add_noise


def create_embeddings() -> List[LangchainEmbedding]:
    """
    Create a list of embeddings using the specified models.

    This function creates embeddings using a list of model names defined in EMBEDDING_MODELS.

    Returns:
        List[Embedding]: A list of embedding objects created from the specified models.
    """
    return [create_langchain_embedding(model_name) for model_name in EMBEDDING_MODELS]


def prepare_llm() -> LLM:
    """
    Initializes and retrieves the configured language model.

    This function configures and retrieves a language model with the specified settings from
    the global settings object. It sets up the maximum number of new tokens, the model's
    temperature, and the context window size for the language model.

    Returns:
        HuggingFaceLLM: An instance of the language model configured with the specified settings.
    """
    return get_zephyr(
        max_new_tokens=settings.llm.num_output,
        model_temperature=settings.llm.temperature,
        context_window=settings.llm.context_window,
    )


async def retrieve_and_evaluate(
    nodes_to_retrieve: List[NodesForTitle],
    nodes_add_noise: List[NodesForTitle],
    embeddings: List[HuggingFaceEmbeddings],
    llm: LLM,
) -> List[Dict[str, Union[str, Any]]]:
    """
    Retrieve and evaluate retrievers for specified nodes and embeddings.

    This function retrieves and evaluates retrievers for a set of nodes to retrieve and nodes to add noise
    using a list of embeddings and a language model. It evaluates retrievers against predefined QA datasets
    using specified metrics and formats the results.

    Args:
        nodes_to_retrieve (List[NodesForTitle]): List of nodes to retrieve.
        nodes_add_noise (List[NodesForTitle]): List of nodes to add noise.
        embeddings (List[HuggingFaceEmbeddings]): List of embedding models to be used.
        llm (LLM): The language model for retrievers.

    Returns:
        List[Dict[str, Union[str, Any]]]: A list of dictionaries containing retrieval evaluation results,
        including details of embeddings and retrievers used.
    """
    metrics = ["mrr", "hit_rate"]
    qa_datasets = load_qa_datasets(EVAL_RETRIEVAL_NODES_QA_APPROVED)

    retrieval_results = []
    for embed_model in embeddings:
        indices, service_ctx = setup_retrieval_context(
            embed_model, nodes_to_retrieve, nodes_add_noise, llm
        )
        retrievers = create_retrievers(indices, service_ctx, llm)

        evaluation_results = await evaluate_retrievers(retrievers, qa_datasets, metrics)

        for retriever in retrievers:
            retrieval_results.extend(
                format_and_extract_results(evaluation_results, embed_model, retriever)
            )

    return retrieval_results


async def evaluate_retrievers(retrievers, qa_datasets, metrics):
    results = []
    for retriever in retrievers:
        evaluation_results = await evaluate_retriever_on_datasets(
            retriever, qa_datasets, metrics
        )
        results.extend(evaluation_results)
    return results


def create_retrievers(indices, service_ctx, llm):
    retriever_factory = RetrieverFactory(indices["vector"], service_ctx, llm)
    return retriever_factory.create_vector_index_retrievers(
        HYPERPARAMETERS_VECTOR_INDEX_RETRIEVER
    )


def setup_retrieval_context(embed_model, nodes_to_retrieve, nodes_add_noise, llm):
    storage_ctx, service_ctx = setup_contexts(settings, llm, embed_model)
    indices = setup_indexing(
        nodes_to_retrieve=nodes_to_retrieve,
        nodes_add_noise=nodes_add_noise,
        storage_context=storage_ctx,
        service_context=service_ctx,
        add_doc_metadata=True,
        index_per_doc_title=False,
    )
    return indices, service_ctx


def format_and_extract_results(evaluation_results, embed_model, retriever):
    formatted_results = []
    for dataset_name, eval_res in evaluation_results:
        result_info = extract_result_info(embed_model, retriever)
        result_info.update(
            {
                "qa_dataset_name": dataset_name,
                **_extract_score_averages(_format_retrieval_eval_results(eval_res)),
            }
        )
        formatted_results.append(result_info)
    return formatted_results


def extract_result_info(
    embed_model: LangchainEmbedding,
    retriever: Union[
        QueryFusionRetriever, VectorIndexRetriever, VectorIndexAutoRetriever
    ],
) -> Dict[str, Optional[Union[int, str, float]]]:
    """
    Extracts the configuration details from embedding models and retrievers.

    Args:
        embed_model: An instance of LangchainEmbedding.
        retriever: An instance of one of the following:
            QueryFusionRetriever, VectorIndexRetriever, VectorIndexAutoRetriever.

    Returns:
        A dictionary with configuration details of the embed_model and retriever.
    """

    def safe_get_attribute(parent, attribute, default=None):
        return (
            getattr(parent, attribute, default)
            if hasattr(parent, attribute)
            else default
        )

    embed_model_name = safe_get_attribute(embed_model, "model_name")
    top_retriever_mode = safe_get_attribute(retriever, "mode")
    sub_retriever = safe_get_attribute(retriever, "_retrievers")

    sub_retriever_mode = (
        safe_get_attribute(sub_retriever[0], "_vector_store_query_mode")
        if sub_retriever
        else None
    )

    similarity_top_k = safe_get_attribute(retriever, "similarity_top_k")
    num_queries = safe_get_attribute(retriever, "num_queries")
    alpha = (
        safe_get_attribute(sub_retriever[0], "_alpha")
        if sub_retriever
        else safe_get_attribute(retriever, "_alpha")
    )
    llm = safe_get_attribute(safe_get_attribute(retriever, "_llm"), "model_name")

    return {
        "embed_model_name": embed_model_name,
        "top_retriever_mode": top_retriever_mode,
        "sub_retriever_mode": sub_retriever_mode,
        "similarity_top_k": similarity_top_k,
        "num_queries": num_queries,
        "alpha": alpha,
        "llm": llm,
    }


# Create a function to create an instance of LangchainEmbedding using a specific set of parameters.
def create_langchain_embedding(model_name: str) -> LangchainEmbedding:
    """create an instance of LangchainEmbedding using model_name"""

    return LangchainEmbedding(
        HuggingFaceEmbeddings(
            model_name=model_name,
            model_kwargs={"device": "cuda"},
            encode_kwargs={"normalize_embeddings": False},
        )
    )


async def gather_with_concurrency(n: int, coros: List[Awaitable]) -> List:
    """
    Limits concurrency when running multiple coroutines using asyncio.

    :param n: Maximum number of tasks to run concurrently.
    :type n: int
    :param coros: A list of coroutine objects to be executed concurrently.
    :type coros: List[Awaitable]
    :return: A list containing the results returned by each coroutine function.
    :rtype: List
    """
    semaphore = asyncio.Semaphore(n)

    async def sem_coro(coro: Awaitable) -> Any:
        async with semaphore:
            return await coro

    return await asyncio.gather(*(sem_coro(c) for c in coros))


async def evaluate_retriever_on_datasets(
    retriever,
    datasets: List[Tuple[str, EmbeddingQAFinetuneDataset]],
    metrics: List[str],
):
    """Asynchronously evaluates retriever on provided Q&A datasets"""

    async def evaluate_single_dataset(name, dataset):
        evaluator = RetrieverEvaluator.from_metric_names(metrics, retriever=retriever)
        evaluations = await evaluator.aevaluate_dataset(dataset)
        return name, evaluations

    coros = [evaluate_single_dataset(name=d[0], dataset=d[1]) for d in datasets]
    evaluation_results = await gather_with_concurrency(10, coros)

    return evaluation_results


class RetrieverFactory:
    def __init__(self, vector_indices, service_ctx, llm):
        self.vector_indices = vector_indices
        self.service_ctx = service_ctx
        self.llm = llm

    def create_fusion_retrievers(
        self, hyperparameters: Dict[str, List[Any]]
    ) -> List[QueryFusionRetriever]:
        """Create list of QueryFusionRetriever for each param set in the param grid"""

        def make_retriever(**kwargs) -> QueryFusionRetriever:
            """Create a list of QueryFusionRetriever objects."""
            vector_retriever = VectorIndexRetriever(
                index=self.vector_indices[0],
                service_context=self.service_ctx,
                similarity_top_k=kwargs.get("similarity_top_k", 2),
                vector_store_query_mode=kwargs.get(
                    "vector_store_query_mode", VectorStoreQueryMode.DEFAULT
                ),
                alpha=kwargs.get("alpha", 0.75),
            )

            # bm25_retriever = BM25Retriever.from_defaults(
            #     index=vector_indices[0], similarity_top_k=similarity_top_k
            # )
            retriever = QueryFusionRetriever(
                retrievers=[vector_retriever],
                llm=self.llm,
                use_async=False,
                verbose=True,
                mode=kwargs.get("mode", FUSION_MODES.SIMPLE),
                similarity_top_k=kwargs.get("similarity_top_k", 2),
                num_queries=kwargs.get("num_queries", 4),
            )
            return retriever

        params = list(itertools.product(*hyperparameters.values()))
        retrievers = [
            make_retriever(**dict(zip(hyperparameters.keys(), param)))
            for param in params
        ]
        return retrievers

    def create_vector_index_retrievers(
        self, hyperparameters: Dict[str, List[Any]]
    ) -> List[VectorIndexRetriever]:
        """Create a list of VectorIndexRetriever objects."""

        def make_retriever(**kwargs) -> VectorIndexRetriever:
            return VectorIndexRetriever(
                index=self.vector_indices[0],
                service_context=self.service_ctx,
                **kwargs,
            )

        params = list(itertools.product(*hyperparameters.values()))
        retrievers = [
            make_retriever(**dict(zip(hyperparameters.keys(), param)))
            for param in params
        ]
        return retrievers

    def create_auto_retrival_retrievers(
        self, hyperparameters: Dict[str, List[Any]]
    ) -> List[VectorIndexAutoRetriever]:
        """Create a list of VectorIndexAutoRetrievers using given hyperparameters."""

        def get_metadata_info():
            authors = [book["author"] for book in metadata_memory.values()]
            titles = [book["title"] for book in metadata_memory.values()]
            return [
                MetadataInfo(
                    name="author",
                    type="str",
                    description=f'Category of the authors, one of {", ".join(authors)}',
                ),
                MetadataInfo(
                    name="title",
                    type="str",
                    description=f'Title of the book, one of {", ".join(titles)}',
                ),
            ]

        def make_retriever(**kwargs) -> VectorIndexAutoRetriever:
            vector_store_info = VectorStoreInfo(
                content_info="Literature/ books on productivity, life-hacks and lifestyle",
                metadata_info=get_metadata_info(),
            )
            return VectorIndexAutoRetriever(
                index=self.vector_indices[0],
                service_context=self.service_ctx,
                vector_store_info=vector_store_info,
                **kwargs,
            )

        params = list(itertools.product(*hyperparameters.values()))
        retrievers = [
            make_retriever(**dict(zip(hyperparameters.keys(), param)))
            for param in params
        ]
        return retrievers


def setup_indexing(
    nodes_to_retrieve: List[NodesForTitle],
    nodes_add_noise: List[NodesForTitle],
    storage_context: StorageContext,
    service_context: ServiceContext,
    add_doc_metadata: bool = False,
    index_per_doc_title: bool = False,
) -> Dict[str, List[Union[VectorStoreIndex, SummaryIndex]]]:
    """Indexing with chosen strategy"""

    def _create_indices_per_doc_title(title_w_nodes: List[NodesForTitle]):
        for title_with_nodes in title_w_nodes:
            if add_doc_metadata:
                metadata_for_title = metadata_memory[title_with_nodes.title]
                for node in title_with_nodes.nodes:
                    node.metadata = metadata_for_title

            vector_index = VectorStoreIndex(
                nodes=title_with_nodes.nodes,
                storage_context=storage_context,
                service_context=service_context,
                show_progress=False,
            )
            vector_index.set_index_id(f"{title_with_nodes.nodes[0].ref_doc_id}")
            summary_index = SummaryIndex(
                nodes=title_with_nodes.nodes,
                storage_context=storage_context,
                service_context=service_context,
                show_progress=False,
            )
            summary_index.set_index_id(f"{title_with_nodes.nodes[0].ref_doc_id}")
            yield (
                vector_index,
                summary_index,
            )

    def _create_indices_at_once(title_w_nodes: List[NodesForTitle]):
        if add_doc_metadata:
            for title_with_nodes in title_w_nodes:
                metadata_for_title = metadata_memory.get(title_with_nodes.title, {})
                for node in title_with_nodes.nodes:
                    node.metadata = metadata_for_title

        all_nodes = sum(
            (title_with_nodes.nodes for title_with_nodes in title_w_nodes), []
        )
        vector_index = VectorStoreIndex(
            nodes=all_nodes,
            storage_context=storage_context,
            service_context=service_context,
            show_progress=False,
        )
        summary_index = SummaryIndex(
            nodes=all_nodes,
            storage_context=storage_context,
            service_context=service_context,
            show_progress=False,
        )
        return [(vector_index, summary_index)]

    creator = (
        _create_indices_per_doc_title
        if index_per_doc_title
        else _create_indices_at_once
    )

    to_retrieve_and_add_noise = nodes_to_retrieve + nodes_add_noise
    indices = {}
    vectors, summaries = zip(*creator(to_retrieve_and_add_noise))
    indices["vector"], indices["summary"] = (
        list(vectors),
        list(summaries),
    )

    return indices


def setup_contexts(
    settings: Settings, llm: LLM, embed_model: LangchainEmbedding
) -> Tuple[StorageContext, ServiceContext]:
    """Setup contexts"""

    storage_ctx = _setup_databases(settings)

    service_ctx = ServiceContext.from_defaults(
        llm=llm, chunk_size=settings.parser.chunk_size, embed_model=embed_model
    )
    return storage_ctx, service_ctx


def _setup_databases(settings: Settings) -> StorageContext:
    """Setup databases with specific collection and database name specific to evaluating retrivers"""

    weaviate_client = weaviate.Client(
        url=f"http://{settings.db_vector.host}:{settings.db_vector.port}"
    )
    mongodb_client = pymongo.MongoClient(
        host=settings.db_no_sql.host, port=settings.db_no_sql.port
    )

    purge_weaviate_schema(weaviate_client, settings.db_vector.collection.name)
    purge_mongo_database(mongodb_client, settings.db_no_sql.database.name)

    storage_context = StorageContext.from_defaults(
        vector_store=WeaviateVectorStore(
            weaviate_client=weaviate_client,
            index_name=settings.db_vector.collection.name,
        ),
        index_store=MongoIndexStore.from_host_and_port(
            host=settings.db_no_sql.host,
            port=settings.db_no_sql.port,
            db_name=settings.db_no_sql.database.name,
            namespace=settings.db_no_sql.collection_index.name,
        ),
        docstore=MongoDocumentStore.from_host_and_port(
            host=settings.db_no_sql.host,
            port=settings.db_no_sql.port,
            db_name=settings.db_no_sql.database.name,
            namespace=settings.db_no_sql.collection_document.name,
        ),
    )
    return storage_context


def create_and_save_nodes_for_all_docs() -> None:
    """Loop to create and save nodes from documents"""

    for documents_path, nodes_path in zip(
        [EVAL_DOCS_TO_FIND, EVAL_DOCS_ADD_NOISE],
        [EVAL_RETRIEVAL_NODES_TO_RETRIEVE, EVAL_RETRIEVAL_NODES_ADD_NOISE],
    ):
        _create_and_save_nodes(
            path_to_save_to=nodes_path,
            path_to_load_from=documents_path,
            chunk_size=settings.parser.chunk_size,
            chunk_overlap=settings.parser.chunk_overlap,
        )


# TODO: refactor to create and save
def _create_and_save_nodes(
    path_to_save_to: Path, path_to_load_from: Path, chunk_size: int, chunk_overlap: int
) -> None:
    """Create llamaindex TextNodes and save them under the path for each document,
    together with its parser metadata.

    Name of the saved json: {title}_{chunk_size}_{chunk_overlap}.json
    """

    path_to_save_to.mkdir(parents=True, exist_ok=True)

    base_node_parser = SimpleNodeParser.from_defaults(
        text_splitter=TokenTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        ),
    )

    any_files_in_dir = list(path_to_load_from.iterdir())
    if not any_files_in_dir:
        logger.info(
            f"There is no file under {str(path_to_load_from)}; hence no nodes to create from this path."
        )
        return

    documents = load_data_from_path(input_dir=path_to_load_from, collect_pages=True)

    for doc in documents:
        base_nodes = base_node_parser.get_nodes_from_documents(
            documents=[doc], show_progress=False
        )
        title = Path(base_nodes[0].ref_doc_id).stem

        path_for_cache_of_nodes_per_doc = path_to_save_to.joinpath(
            BASE_NODES_FNAME_PATTERN.format(
                title=title,
                split_pattern=TITLE_METADATA_SPLIT_PATTERN,
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
            )
        )
        if not path_for_cache_of_nodes_per_doc.exists():
            _save_base_nodes(base_nodes, path_for_cache_of_nodes_per_doc)
        else:
            logger.info(
                f"The file {path_for_cache_of_nodes_per_doc} exists already, if you would like to recreate it, "
                "please delete the file manually first. You will afterwards need to required the QA dataset"
            )


def _save_base_nodes(base_nodes: List[TextNode], path: Path):
    """Save list of base nodes to json, so they can be read again"""

    base_nodes_jsons = [node.to_json() for node in base_nodes]
    with open(path, "w") as output_file:
        for m in base_nodes_jsons:
            output_file.write(json.dumps(m) + "\n")


def load_nodes_with_title_to_retrieve(path: Path) -> List[NodesForTitle]:
    """Loop to get all nodes with their respective title for all documents on the retrieval path"""

    nodes_w_title_to_retrieve = [
        title_with_nodes
        for title_with_nodes in _load_nodes_with_title(path_to_base_nodes=path)
    ]

    return nodes_w_title_to_retrieve


def _load_nodes_with_title(
    path_to_base_nodes: Path
) -> Generator[NodesForTitle, None, None]:
    """For a document title, load the nodes and the parser metadata from the filename"""

    for base_node_file in path_to_base_nodes.glob("*.json"):
        ## TODO: improve the splliting of the filename to an object or something more robust.
        fname_splitted = base_node_file.stem.split(TITLE_METADATA_SPLIT_PATTERN)
        doc_title = fname_splitted[0]
        parser_metadata = fname_splitted[1].split("_")
        base_nodes = _load_base_nodes(base_node_file)
        yield NodesForTitle(
            title=doc_title,
            nodes=base_nodes,
            chunk_size=parser_metadata[0],
            chunk_overlap=parser_metadata[1],
        )


def _load_base_nodes(path) -> List[TextNode]:
    """Load textnode(s) from json path to TextNode Objects"""

    with open(path, "r") as fp:
        json_list = list(fp)
        base_nodes = [TextNode.from_json(json.loads(z)) for z in json_list]
        return base_nodes


def generate_and_save_qa_dataset(
    titles_with_nodes: List[NodesForTitle], path: Path, llm: LLM
):
    """Genereate a question conext pairs for nodes"""

    path.mkdir(parents=True, exist_ok=True)

    for nodes_obj in titles_with_nodes:
        path_qa_dataset = path.joinpath(
            BASE_NODES_FNAME_PATTERN.format(
                title=nodes_obj.title,
                split_pattern=TITLE_METADATA_SPLIT_PATTERN,
                chunk_size=nodes_obj.chunk_size,
                chunk_overlap=nodes_obj.chunk_overlap,
            )
        )

        if path_qa_dataset.exists():
            logger.info(f"QA dataset was already generated for {str(path_qa_dataset)}")
            continue

        qa_dataset = generate_question_context_pairs(
            nodes_obj.nodes, llm=llm, num_questions_per_chunk=2
        )
        qa_dataset.save_json(path=path_qa_dataset)


def load_qa_datasets(
    path_to_evaluation_datasets: Path
) -> List[Tuple[str, EmbeddingQAFinetuneDataset]]:
    """Loads all QA datasets from given parent path as dictionary."""

    def load_qa_dataset(file_path: Path) -> Tuple[str, EmbeddingQAFinetuneDataset]:
        data = EmbeddingQAFinetuneDataset.from_json(file_path)
        dataset_name = file_path.stem
        return (dataset_name, data)

    # Load each dataset from path using map function
    loaded_datasets = list(
        map(load_qa_dataset, path_to_evaluation_datasets.glob("*.json"))
    )
    return loaded_datasets


def _format_retrieval_eval_results(
    retrieval_eval_results: List[RetrievalEvalResult]
) -> Dict[str, Dict[str, float]]:
    """Format the RetrievalEvalResult set."""

    all_metrics_results = {}
    metric_names = list(retrieval_eval_results[0].metric_vals_dict.keys())
    for metric_name in metric_names:
        metric_results = {}
        metric_results["scores"] = [
            _eval.metric_vals_dict.get(metric_name) for _eval in retrieval_eval_results
        ]
        metric_results["score_avg"] = _calculate_average(metric_results["scores"])
        all_metrics_results[metric_name] = metric_results
    return all_metrics_results


def _calculate_average(values: List[Union[float, int]]) -> float:
    """Calculates the average of a list of numerical values.
    Args:
        values (List[Union[float, int]): List of numeric values.

    Returns:
        float: The average value.

    Examples:
        >>> calculate_average([10, 20, 30, 40, 50])
        30.0
    """
    total = sum(values)
    length = len(values)
    if length == 0:
        raise ValueError("Cannot compute average for empty list")
    return round(total / length, 2)


def _extract_score_averages(data: Dict[str, Dict[str, float]]) -> Dict[str, float]:
    """Extracts score averages from a given data structure.

    Args:
        data (list[dict]): A list of dictionaries containing metric values and their average scores.
            Each dictionary has two keys: 'metric' and 'score_avg'. The value corresponding to
            the 'metric' key is a string, and the value corresponding to 'score_avg' is a float.

    Returns:
        dict: A dictionary with keys as the metrics and their average scores as value.
    """
    return {k: v["score_avg"] for k, v in data.items() if "score_avg" in v}


if __name__ == "__main__":
    asyncio.run(retrieval_evaluation_and_report())
