"""Configuration for Docsorcerer."""

from functools import lru_cache
from pathlib import Path
from typing import Optional

from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings, SettingsConfigDict

# Path where your env should be placed:
# /home/<user>/DocSorcerer/.env
DOT_ENV_PATH = Path.cwd().joinpath(".env")


# Data handling related settings
class Parser(BaseModel):
    """Document/Node/Text parsing configurations"""

    separator: str = Field(default=" ")
    chunk_size: int = Field(default=1024)
    chunk_overlap: int = Field(default=20)


class EmbedSettings(BaseModel):
    name: str = Field(default="BAAI/bge-small-en-v1.5")


# LLM and prompting related settings
class LLMSettings(BaseModel):
    context_window: int = Field(default=3800)
    num_output: int = Field(default=256)
    temperature: float = Field(default=0.1)


class PromptHelperSettings(BaseModel):
    context_window: int = Field(default=3900)
    num_output: int = Field(default=256)
    chunk_overlap_ratio: float = Field(default=0.1)


## Database related settings
class DbSettings(BaseModel):
    host: str = Field(default="localhost")
    port: int
    name: Optional[str] = Field(description="Name of the database type")


class Collection(BaseModel):
    name: str = Field(
        description="Name of the collection (equivalent of an RDBMS table, within single database in MongoDB)/ "
        "class (weaviate) for objects to be stored"
    )


class VectorDbSettings(DbSettings):
    collection: Collection


class Database(BaseModel):
    name: str = Field(
        description="Name of the database that contains collections (equivalent of an RDBMS table)"
    )


class NoSqlDbSettings(DbSettings):
    database: Database
    collection_index: Collection
    collection_document: Collection


class Settings(BaseSettings, case_sensitive=False):
    """Configuration for docsorcerer"""

    model_config = SettingsConfigDict(
        env_file=DOT_ENV_PATH,
        env_file_encoding="utf-8",
        extra="ignore",
        env_nested_delimiter="__",
    )

    parser: Parser
    embed: EmbedSettings
    llm: LLMSettings
    prompt_helper: PromptHelperSettings
    db_vector: VectorDbSettings
    db_no_sql: NoSqlDbSettings


@lru_cache()
def get_api_settings():
    """Get pydantic settings"""
    return Settings()
