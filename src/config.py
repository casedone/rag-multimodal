"""
Configuration file for RAG Multimodal application.
Contains all the configuration parameters used across the application.

Usage:
    ```python
    from src.config import config
    
    # Access configuration values by path
    model = config.get("model", "text_generation", default="gpt-4.1-mini")
    embed_model = config.get("model", "embeddings", default="text-embedding-3-small")
    
    # Get specialized configuration objects
    milvus_args = config.get_milvus_connection_args()
    pdf_options = config.get_pdf_pipeline_options()
    
    # Use custom configuration file
    from src.config import ConfigLoader
    custom_config = ConfigLoader("path/to/custom/config.yaml")
    custom_model = custom_config.get("model", "text_generation")
    
    # Strict mode - prevent accidental new key creation
    strict_config = ConfigLoader(allow_new_keys=False)
    strict_config.set("model", "text_generation", "gpt-4")  # OK - existing key
    # strict_config.set("new", "key", "value")  # Would raise KeyError
    strict_config.set("new", "key", "value", force=True)  # OK - forced
    ```
"""
import os
import yaml
from pathlib import Path
from typing import Dict, Any, Optional, Union
from dotenv import load_dotenv

class ConfigLoader:
    """Configuration loader for RAG Multimodal application"""
    
    def __init__(self, config_path: Optional[Union[str, Path]] = None, allow_new_keys: bool = True):
        """Initialize ConfigLoader with optional custom config path
        
        Args:
            config_path: Optional path to configuration file
            allow_new_keys: Whether to allow setting keys that don't exist in the original config
        """
        self._config = None
        self._allow_new_keys = allow_new_keys
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
    
    def set(self, *keys_and_value, force: bool = False) -> None:
        """Set a configuration value by key path
        
        Args:
            *keys_and_value: Key path followed by the value to set
                           Last argument is the value, preceding arguments are the key path
            force: If True, allows setting new keys even when allow_new_keys is False
            
        Example:
            config.set("model", "text_generation", "gpt-4")
            config.set("database", "uri", "http://localhost:19530")
            config.set("new", "key", "value", force=True)  # Force creation of new key
            
        Raises:
            KeyError: If trying to set a non-existing key when allow_new_keys is False
        """
        if len(keys_and_value) < 2:
            raise ValueError("At least one key and a value must be provided")
            
        # Split keys and value
        keys = keys_and_value[:-1]
        value = keys_and_value[-1]
            
        # Initialize config if None
        if self._config is None:
            self._config = {}
            
        # Check if we're allowed to create new keys
        if not self._allow_new_keys and not force:
            self._validate_key_exists(keys)
            
        # Navigate to the parent of the target key
        current = self._config
        for key in keys[:-1]:
            if key not in current:
                if not self._allow_new_keys and not force:
                    raise KeyError(f"Key path {'.'.join(keys)} does not exist and allow_new_keys is False. Use force=True to override.")
                current[key] = {}
            elif not isinstance(current[key], dict):
                # Convert non-dict values to dict to allow nested setting
                current[key] = {}
            current = current[key]
            
        # Check final key
        if keys[-1] not in current and not self._allow_new_keys and not force:
            raise KeyError(f"Key '{keys[-1]}' does not exist in path {'.'.join(keys[:-1])} and allow_new_keys is False. Use force=True to override.")
            
        # Set the final value
        current[keys[-1]] = value
    
    def _validate_key_exists(self, keys) -> None:
        """Validate that a key path exists in the configuration
        
        Args:
            keys: Key path to validate
            
        Raises:
            KeyError: If the key path doesn't exist
        """
        current = self._config
        for i, key in enumerate(keys):
            if not isinstance(current, dict) or key not in current:
                key_path = '.'.join(keys[:i+1])
                raise KeyError(f"Key path '{key_path}' does not exist and allow_new_keys is False. Use force=True to override.")
            current = current[key]
    
    def reload(self, config_path: Optional[Union[str, Path]] = None) -> None:
        """Reload configuration from file"""
        self._load_config(config_path)
    
    def set_allow_new_keys(self, allow: bool) -> None:
        """Set whether new keys are allowed to be created
        
        Args:
            allow: Whether to allow creation of new keys
        """
        self._allow_new_keys = allow
    
    def get_allow_new_keys(self) -> bool:
        """Get whether new keys are allowed to be created
        
        Returns:
            bool: Current allow_new_keys setting
        """
        return self._allow_new_keys
    
    def get_milvus_connection_args(self) -> Dict[str, str]:
        """Get Milvus connection arguments"""
        return {
            "uri": self.get("database", "uri", default="http://localhost:19530"),
            "token": self.get("database", "token", default="root:Milvus"),
            "db_name": self.get("database", "name", default="rag_multimodal")
        }
    
    def get_pdf_pipeline_options(self) -> Dict[str, Any]:
        """Get PDF pipeline options"""
        from docling.datamodel.pipeline_options import (
            PdfPipelineOptions,
            PictureDescriptionApiOptions
        )
        
        # Get API configuration
        openai_api_key = os.environ.get("OPENAI_API_KEY", "")
        model_url = self.get("model", "url", default="https://api.openai.com/v1/chat/completions")
        model_name = self.get("model", "text_generation", default="gpt-4.1-mini")
        model_timeout = self.get("model", "timeout", default=60)
        picture_prompt = self.get("document", "picture_description", "prompt_picture_description", 
                                default="Describe this image in sentences in a single paragraph.")
        image_scale = self.get("document", "image_resolution_scale", default=2)
        
        picture_desc_api_option = PictureDescriptionApiOptions(
            url=model_url,
            prompt=picture_prompt,
            params={"model": model_name},
            headers={"Authorization": f"Bearer {openai_api_key}"},
            timeout=model_timeout,
        )
        
        return PdfPipelineOptions(
            images_scale=image_scale,
            generate_picture_images=True,
            do_picture_description=True,
            picture_description_options=picture_desc_api_option,
            enable_remote_services=True,  # to access remote API
        )

# Create a default configuration instance
config = ConfigLoader()
