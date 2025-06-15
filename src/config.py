"""Configuration management for the RAG system."""
import os
import yaml
from pathlib import Path
from typing import Dict, List, Any
from dataclasses import dataclass


@dataclass
class DataConfig:
    source_dir: str
    file_extensions: List[str]
    company_name: str
    dataset_description: str


@dataclass
class OllamaConfig:
    base_url: str
    embedding_model: str
    chat_model: str
    timeout: int


@dataclass
class ProcessingConfig:
    chunk_size: int
    chunk_overlap: int
    min_chunk_size: int
    max_retrieved_chunks: int


@dataclass
class VectorDBConfig:
    type: str
    path: str
    collection_name: str
    distance_metric: str


@dataclass
class CLIConfig:
    history_size: int
    show_sources: bool
    max_sources: int


@dataclass
class LoggingConfig:
    level: str
    file: str


@dataclass
class Config:
    data: DataConfig
    ollama: OllamaConfig
    processing: ProcessingConfig
    vector_db: VectorDBConfig
    cli: CLIConfig
    logging: LoggingConfig


def load_config(config_path: str = "config.yaml") -> Config:
    """Load configuration from YAML file."""
    if not os.path.exists(config_path):
        raise FileNotFoundError(
            f"Configuration file not found: {config_path}\n"
            f"Please copy config.example.yaml to {config_path} and customize it."
        )
    
    with open(config_path, 'r') as f:
        config_dict = yaml.safe_load(f)
    
    return Config(
        data=DataConfig(**config_dict['data']),
        ollama=OllamaConfig(**config_dict['ollama']),
        processing=ProcessingConfig(**config_dict['processing']),
        vector_db=VectorDBConfig(**config_dict['vector_db']),
        cli=CLIConfig(**config_dict['cli']),
        logging=LoggingConfig(**config_dict['logging'])
    )


def get_absolute_path(relative_path: str, base_path: str = None) -> str:
    """Convert relative path to absolute path."""
    if base_path is None:
        base_path = Path(__file__).parent.parent
    
    if os.path.isabs(relative_path):
        return relative_path
    
    return str(Path(base_path) / relative_path)