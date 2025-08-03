# RAG Multimodal

A multimodal Retrieval-Augmented Generation (RAG) system that processes text and images from documents.

## Configuration

The application uses a simplified, flexible configuration system based on the `ConfigLoader` class:

### Configuration Files

- **config.yaml**: Central configuration file containing all settings in YAML format
- **src/config.py**: Python module that provides the `ConfigLoader` class and global `config` instance

### Configuration Structure

The configuration is organized into these main sections:

- **Model Configuration**: LLM settings for text generation and embeddings
- **Document Processing**: Settings for handling documents and images
- **Vector Database**: Milvus connection and database settings
- **Retrieval Configuration**: Parameters for the retrieval process

### Accessing Configuration

The system provides a simple, consistent API for accessing configuration values:

**Basic Usage (Recommended)**:
```python
from src.config import config

# Access configuration values by path
model = config.get("model", "text_generation", default="gpt-4.1-mini")
embed_model = config.get("model", "embeddings", default="text-embedding-3-small")
max_tokens = config.get("document", "max_tokens", default=512)

# Get specialized configuration objects
milvus_args = config.get_milvus_connection_args()
pdf_options = config.get_pdf_pipeline_options()
```

**Setting Configuration Values**:
```python
# Set configuration values at runtime
config.set("database", "collection_name", "my_custom_collection")
config.set("model", "text_generation", "gpt-4")

# Control whether new keys can be created
config.set_allow_new_keys(False)  # Prevent accidental new keys
config.set("new", "key", "value", force=True)  # Force creation when needed
```

**Custom Configuration Files**:
```python
from src.config import ConfigLoader

# Use a custom configuration file
custom_config = ConfigLoader("path/to/custom/config.yaml")
value = custom_config.get("database", "uri")

# Strict mode - prevent new key creation
strict_config = ConfigLoader(allow_new_keys=False)
strict_config.set("model", "text_generation", "gpt-4")  # OK - existing key
# strict_config.set("new", "key", "value")  # Would raise KeyError
```

### Configuration Safety

The configuration system includes several safety features:

**Default Values**: All configuration access includes sensible defaults:
```python
# If "model.text_generation" doesn't exist, uses "gpt-4.1-mini"
model = config.get("model", "text_generation", default="gpt-4.1-mini")
```

**Key Validation**: Control whether new configuration keys can be created:
```python
# Prevent accidental typos from creating new keys
strict_config = ConfigLoader(allow_new_keys=False)

# This would raise KeyError if "typo_section" doesn't exist
# strict_config.set("typo_section", "key", "value")

# Use force=True when you really need to create new keys
strict_config.set("new_section", "key", "value", force=True)
```

**Runtime Configuration**: Change settings without restarting:
```python
# Toggle key creation policy at runtime
config.set_allow_new_keys(False)  # Enable strict mode
config.set_allow_new_keys(True)   # Allow new keys again
```

### Environment Variables

Sensitive information like API keys should be stored in a `.env` file:

```
OPENAI_API_KEY=your_api_key_here
```

The configuration system will load these environment variables automatically.

### Recent Configuration Improvements

The configuration system has been recently simplified and improved:

**✅ Simplified Architecture**:
- Removed complex `Config` class with multiple access patterns
- Single `ConfigLoader` class provides consistent API
- Global `config` instance for convenient access

**✅ Enhanced Safety**:
- `allow_new_keys` parameter prevents accidental key creation
- `force` parameter for intentional new key creation
- Runtime control over key creation policy

**✅ Better Modularity**:
- Core components (like `MilvusStore`) no longer depend on config objects
- Configuration values extracted at orchestration level
- Explicit parameter passing maintains clean architecture

**✅ Improved Developer Experience**:
- Consistent `config.get("section", "key", default=value)` pattern
- Clear error messages with helpful suggestions
- Support for custom configuration files
- Runtime configuration changes without restart