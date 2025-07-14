"""
Configuration file for RAG Multimodal application.
Contains all the configuration parameters used across the application.

Usage:
    1. Direct import of configuration values:
       ```python
       from src.config import MODEL, EMBED_MODEL, MAX_TOKENS
       
       # Use the configuration values directly
       print(f"Using model: {MODEL}")
       ```
       
    2. Using the Config class for grouped access:
       ```python
       from src.config import Config
       
       # Access configuration values as class attributes
       model = Config.MODEL
       embed_model = Config.EMBED_MODEL
       
       # Get specialized configuration objects
       milvus_args = Config.get_milvus_connection_args()
       pdf_options = Config.get_pdf_pipeline_options()
       ```
       
    3. Using the ConfigLoader for direct YAML access:
       ```python
       from src.config import ConfigLoader
       
       # Get the singleton instance
       config_loader = ConfigLoader()
       
       # Access configuration values by path
       value = config_loader.get("database", "uri")
       custom_value = config_loader.get("custom", "section", "key", default="default_value")
       ```
"""
import os
import yaml
from pathlib import Path
from typing import Dict, List, Any, Optional
from dotenv import load_dotenv

class ConfigLoader:
    """Configuration loader for RAG Multimodal application"""
    
    _instance = None
    _config = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(ConfigLoader, cls).__new__(cls)
            cls._instance._load_config()
        return cls._instance
    
    def _load_config(self):
        """Load configuration from YAML file and environment variables"""
        # Load environment variables
        env_path = Path(__file__).parent.parent / ".env"
        load_dotenv(env_path)
        
        # Load YAML configuration
        yaml_path = Path(__file__).parent.parent / "config.yaml"
        if yaml_path.exists():
            with open(yaml_path, 'r') as file:
                self._config = yaml.safe_load(file)
        else:
            raise FileNotFoundError(f"Configuration file not found: {yaml_path}")
    
    @property
    def config(self):
        """Get the loaded configuration"""
        return self._config
    
    def get(self, *keys, default=None):
        """Get a configuration value by key path"""
        value = self._config
        for key in keys:
            if isinstance(value, dict) and key in value:
                value = value[key]
            else:
                return default
        return value

class Config:
    """Configuration class for RAG Multimodal application"""
    
    # Load configuration
    _loader = ConfigLoader()
    
    # API Keys
    OPENAI_API_KEY: str = os.environ.get("OPENAI_API_KEY", "")
    
    # Model Configuration
    MODEL: str = _loader.get("model", "text_generation", default="gpt-4.1-mini")
    EMBED_MODEL: str = _loader.get("model", "embeddings", default="text-embedding-3-small")
    EMBED_MODEL_ID: str = _loader.get("model", "tokenizer", default="sentence-transformers/all-MiniLM-L6-v2")
    MODEL_URL: str = _loader.get("model", "url", default="https://api.openai.com/v1/chat/completions")
    MODEL_TIMEOUT: int = _loader.get("model", "timeout", default=60)
    
    # Document Processing
    IMAGE_RESOLUTION_SCALE: int = _loader.get("document", "image_resolution_scale", default=2)
    MAX_TOKENS: int = _loader.get("document", "max_tokens", default=512)
    DOC_DIR: str = _loader.get("document", "doc_dir", default="doc_sources")
    SUPPORTED_FILE_TYPES: List[str] = _loader.get("document", "supported_file_types", default=[".pdf"])
    PICTURE_DESC_PROMPT: str = _loader.get("document", "picture_description", "prompt_picture_description", default="Describe this image in sentences in a single paragraph.")
    
    # Vector Database Configuration
    URI: str = _loader.get("database", "uri", default="http://localhost:19530")
    DB_NAME: str = _loader.get("database", "name", default="rag_multimodal")
    DB_COLLECTION_NAME: str = _loader.get("database", "collection_name", default="collection_demo")
    NAMESPACE: str = _loader.get("database", "namespace", default="CaseDoneDemo")
    
    # Retrieval Configuration
    RETRIEVAL_K: int = _loader.get("retrieval", "k", default=2)
    RETRIEVAL_WEIGHTS: List[float] = _loader.get("retrieval", "weights", default=[0.6, 0.4])
    
    @classmethod
    def get_picture_desc_api_options(cls) -> Dict[str, Any]:
        """Get picture description API options"""
        return {
            "url": cls.MODEL_URL,
            "prompt": cls.PICTURE_DESC_PROMPT,
            "params": {"model": cls.MODEL},
            "headers": {"Authorization": f"Bearer {cls.OPENAI_API_KEY}"},
            "timeout": cls.MODEL_TIMEOUT,
        }
    
    @classmethod
    def get_pdf_pipeline_options(cls) -> Dict[str, Any]:
        """Get PDF pipeline options"""
        from docling.datamodel.pipeline_options import (
            PdfPipelineOptions,
            PictureDescriptionApiOptions
        )
        
        picture_desc_api_option = PictureDescriptionApiOptions(
            url=cls.get_picture_desc_api_options()["url"],
            prompt=cls.get_picture_desc_api_options()["prompt"],
            params=cls.get_picture_desc_api_options()["params"],
            headers=cls.get_picture_desc_api_options()["headers"],
            timeout=cls.get_picture_desc_api_options()["timeout"],
        )
        
        return PdfPipelineOptions(
            images_scale=cls.IMAGE_RESOLUTION_SCALE,
            generate_picture_images=True,
            do_picture_description=True,
            picture_description_options=picture_desc_api_option,
            enable_remote_services=True,  # to access remote API
        )
    
    @classmethod
    def get_milvus_connection_args(cls) -> Dict[str, str]:
        """Get Milvus connection arguments"""
        return {
            "uri": cls.URI,
            "token": "root:Milvus",
            "db_name": cls.DB_NAME
        }


# For backward compatibility, expose variables at module level
OPENAI_API_KEY = Config.OPENAI_API_KEY
MODEL = Config.MODEL
EMBED_MODEL = Config.EMBED_MODEL
EMBED_MODEL_ID = Config.EMBED_MODEL_ID
MODEL_URL = Config.MODEL_URL
MODEL_TIMEOUT = Config.MODEL_TIMEOUT
IMAGE_RESOLUTION_SCALE = Config.IMAGE_RESOLUTION_SCALE
MAX_TOKENS = Config.MAX_TOKENS
PICTURE_DESC_PROMPT = Config.PICTURE_DESC_PROMPT
URI = Config.URI
DB_NAME = Config.DB_NAME
DB_COLLECTION_NAME = Config.DB_COLLECTION_NAME
NAMESPACE = Config.NAMESPACE
DOC_DIR = Config.DOC_DIR
SUPPORTED_FILE_TYPES = Config.SUPPORTED_FILE_TYPES
RETRIEVAL_K = Config.RETRIEVAL_K
RETRIEVAL_WEIGHTS = Config.RETRIEVAL_WEIGHTS
