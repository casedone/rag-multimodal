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
       
       # Reload configuration from a custom file
       Config("path/to/custom/config.yaml")
       
       # Access updated configuration values
       updated_model = Config.MODEL
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
from typing import Dict, List, Any, Optional, Union
from dotenv import load_dotenv
from functools import lru_cache

class ConfigLoader:
    """Configuration loader for RAG Multimodal application"""
    
    def __init__(self, config_path: Optional[Union[str, Path]] = None):
        """Initialize ConfigLoader with optional custom config path"""
        self._config = None
        self._load_config(config_path)
    
    def _load_config(self, config_path: Optional[Union[str, Path]] = None):
        """Load configuration from YAML file and environment variables"""
        # Load environment variables
        env_path = Path(__file__).parent.parent / ".env"
        load_dotenv(env_path)
        
        # Determine config file path
        if config_path:
            yaml_path = Path(config_path)
        else:
            yaml_path = Path(__file__).parent.parent / "config.yaml"
            
        if yaml_path.exists():
            with open(yaml_path, 'r') as file:
                self._config = yaml.safe_load(file)
        else:
            raise FileNotFoundError(f"Configuration file not found: {yaml_path}")
    
    @property
    def config(self) -> Dict[str, Any]:
        """Get the loaded configuration"""
        return self._config or {}
    
    def get(self, *keys, default=None) -> Any:
        """Get a configuration value by key path"""
        value = self._config
        for key in keys:
            if isinstance(value, dict) and key in value:
                value = value[key]
            else:
                return default
        return value
    
    def reload(self, config_path: Optional[Union[str, Path]] = None) -> None:
        """Reload configuration from file"""
        self._load_config(config_path)

class Config:
    """Configuration class for RAG Multimodal application
    
    This class can be called with a config filepath to reload configurations:
    ```python
    from src.config import Config
    
    # Reload configuration from a custom file
    Config.reload("path/to/custom/config.yaml")
    
    # Access updated configuration values
    model = Config.MODEL
    ```
    """
    
    # Configuration loader instance
    _loader = ConfigLoader()
    
    # Configuration schema with defaults
    _CONFIG_SCHEMA = {
        # API Keys (from environment)
        'OPENAI_API_KEY': lambda: os.environ.get("OPENAI_API_KEY", ""),
        
        # Model Configuration
        'MODEL': ('model', 'text_generation', "gpt-4.1-mini"),
        'EMBED_MODEL': ('model', 'embeddings', "text-embedding-3-small"),
        'EMBED_MODEL_ID': ('model', 'tokenizer', "sentence-transformers/all-MiniLM-L6-v2"),
        'MODEL_URL': ('model', 'url', "https://api.openai.com/v1/chat/completions"),
        'MODEL_TIMEOUT': ('model', 'timeout', 60),
        
        # Document Processing
        'IMAGE_RESOLUTION_SCALE': ('document', 'image_resolution_scale', 2),
        'MAX_TOKENS': ('document', 'max_tokens', 512),
        'DOC_DIR': ('document', 'doc_dir', "doc_sources"),
        'SUPPORTED_FILE_TYPES': ('document', 'supported_file_types', [".pdf"]),
        'PICTURE_DESC_PROMPT': ('document', 'picture_description', 'prompt_picture_description', 
                               "Describe this image in sentences in a single paragraph."),
        
        # Vector Database Configuration
        'URI': ('database', 'uri', "http://localhost:19530"),
        'DB_NAME': ('database', 'name', "rag_multimodal"),
        'DB_COLLECTION_NAME': ('database', 'collection_name', "collection_demo"),
        'NAMESPACE': ('database', 'namespace', "CaseDoneDemo"),
        'MILVUS_TOKEN': ('database', 'token', "root:Milvus"),
        
        # Retrieval Configuration
        'RETRIEVAL_K': ('retrieval', 'k', 2),
        'RETRIEVAL_WEIGHTS': ('retrieval', 'weights', [0.6, 0.4]),
    }
    
    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        cls._load_attributes()
    
    @classmethod
    def _load_attributes(cls) -> None:
        """Load all configuration attributes from schema"""
        for attr_name, config_def in cls._CONFIG_SCHEMA.items():
            if callable(config_def):  # Environment variable
                value = config_def()
            else:  # YAML configuration
                *keys, default = config_def
                value = cls._loader.get(*keys, default=default)
            setattr(cls, attr_name, value)
    
    @classmethod
    @lru_cache(maxsize=1)
    def get_picture_desc_api_options(cls) -> Dict[str, Any]:
        """Get picture description API options (cached)"""
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
        
        # Get options once and reuse
        api_options = cls.get_picture_desc_api_options()
        picture_desc_api_option = PictureDescriptionApiOptions(
            url=api_options["url"],
            prompt=api_options["prompt"],
            params=api_options["params"],
            headers=api_options["headers"],
            timeout=api_options["timeout"],
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
            "token": cls.MILVUS_TOKEN,
            "db_name": cls.DB_NAME
        }
    
    @classmethod
    def reload(cls, config_filepath: Optional[str] = None) -> None:
        """Reload configuration from the specified file path
        
        Args:
            config_filepath: Optional path to the configuration YAML file to reload from
        """
        # Clear cache for get_picture_desc_api_options
        cls.get_picture_desc_api_options.cache_clear()
        
        if config_filepath is not None:
            # Validate file path
            config_path = Path(config_filepath)
            if not config_path.exists():
                raise FileNotFoundError(f"Configuration file not found: {config_filepath}")
                
            # Reload the configuration loader with new file
            cls._loader.reload(config_filepath)
        else:
            # Reload with default config file
            cls._loader.reload()
            
        # Refresh all attributes
        cls._load_attributes()
        
        # Update module-level variables for backward compatibility
        _update_module_variables()


def _update_module_variables():
    """Update module-level variables for backward compatibility"""
    # Get current module
    import sys
    current_module = sys.modules[__name__]
    
    # Update all attributes dynamically
    for attr_name in Config._CONFIG_SCHEMA.keys():
        setattr(current_module, attr_name, getattr(Config, attr_name))

# Initialize Config class attributes and module-level variables
Config._load_attributes()
_update_module_variables()
