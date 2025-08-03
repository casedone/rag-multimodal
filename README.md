# RAG Multimodal

A multimodal Retrieval-Augmented Generation (RAG) system that processes text and images from documents.

## Configuration

The application uses a flexible configuration system with multiple access methods:

### Configuration Files

- **config.yaml**: Central configuration file containing all settings in YAML format
- **src/config.py**: Python module that loads and provides access to configuration values

### Configuration Structure

The configuration is organized into these main sections:

- **Model Configuration**: LLM settings for text generation and embeddings
- **Document Processing**: Settings for handling documents and images
- **Vector Database**: Milvus connection and database settings
- **Retrieval Configuration**: Parameters for the retrieval process

### Accessing Configuration

The system provides three ways to access configuration values:

1. **Direct import of module-level variables**:
   ```python
   from src.config import MODEL, EMBED_MODEL, MAX_TOKENS
   
   print(f"Using model: {MODEL}")
   ```

2. **Using the Config class for grouped access**:
   ```python
   from src.config import Config
   
   model = Config.MODEL
   embed_model = Config.EMBED_MODEL
   
   # Get specialized configuration objects
   milvus_args = Config.get_milvus_connection_args()
   pdf_options = Config.get_pdf_pipeline_options()
   ```

3. **Using the ConfigLoader for direct YAML access**:
   ```python
   from src.config import ConfigLoader
   
   config_loader = ConfigLoader()
   value = config_loader.get("database", "uri")
   ```

### Default Values

The configuration system includes default values in the `Config` class that serve several purposes:

- **Fallback Mechanism**: Ensures the application runs even if configuration keys are missing
- **Self-Documentation**: Shows developers what values to expect
- **Type Safety**: Establishes expected data types for configuration parameters
- **Resilience**: Protects against configuration file changes or missing keys
- **Development**: Allows running the code without needing to modify configuration files

### Environment Variables

Sensitive information like API keys should be stored in a `.env` file:

```
OPENAI_API_KEY=your_api_key_here
```

The configuration system will load these environment variables automatically.