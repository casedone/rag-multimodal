# RAG Multimodal

A multimodal Retrieval-Augmented Generation (RAG) system that processes text and images from documents using advanced agentic workflows and multiple vector stores.

## Features

- **Agentic RAG**: LangGraph-based intelligent document retrieval and generation
- **Multiple Vector Stores**: Support for multiple specialized vector stores with dynamic management
- **Multimodal Processing**: Handle text and images from various document formats
- **Flexible Configuration**: Robust configuration system with safety features
- **Gradio UI**: Interactive web interface for easy interaction
- **Comprehensive Logging**: Configurable logging across all modules

## Architecture

The system consists of five main modules:

- **`src/agent.py`**: Agentic RAG implementation using LangGraph
- **`src/milvus_store.py`**: Vector database operations and management
- **`src/index.py`**: Document processing and indexing pipeline
- **`src/config.py`**: Configuration management system
- **`src/ui.py`**: Gradio-based web user interface

## Quick Start

### Basic Usage

```python
from src.agent import AgenticRAG
from src.milvus_store import MilvusStore
from src.index import process_and_index_directory

# 1. Index documents
process_and_index_directory("path/to/documents")

# 2. Create RAG agent
milvus_store = MilvusStore()
rag_agent = AgenticRAG(vector_store=milvus_store)

# 3. Query the system
response = rag_agent.run("What is the main topic of the documents?")
print(response)
```

### Multiple Vector Stores

```python
from src.agent import AgenticRAG
from src.milvus_store import MilvusStore

# Create specialized vector stores
docs_store = MilvusStore(collection_name="documents")
code_store = MilvusStore(collection_name="code_base")

# Configure multiple stores
vector_stores = [
    {
        'store': docs_store,
        'name': 'search_documents',
        'description': 'Search and retrieve information from document collection.',
        'k': 2
    },
    {
        'store': code_store,
        'name': 'search_code',
        'description': 'Search and retrieve information from code repository.',
        'k': 3,
        'ranker_weights': [0.7, 0.3]
    }
]

# Create agent with multiple stores
rag_agent = AgenticRAG(vector_stores=vector_stores)
response = rag_agent.run("Find code examples for authentication")
```

### Web Interface

```python
from src.ui import GradioUI
from src.agent import AgenticRAG
from src.milvus_store import MilvusStore

# Create agent
milvus_store = MilvusStore()
rag_agent = AgenticRAG(vector_store=milvus_store)

# Launch web interface
ui = GradioUI(rag_agent, rag_agent.get_config())
ui.launch(share=True)
```

## Configuration

The application uses a flexible configuration system based on the `ConfigLoader` class:

### Configuration Files

- **`config.yaml`**: Central configuration file containing all settings
- **`src/config.py`**: Configuration management module

### Configuration Structure

```yaml
# Model Configuration
model:
  text_generation: "gpt-4.1-mini"
  embeddings: "text-embedding-3-small"
  tokenizer: "sentence-transformers/all-MiniLM-L6-v2"

# Document Processing
document:
  max_tokens: 512
  supported_file_types: [".pdf"]
  image_resolution_scale: 2

# Vector Database
database:
  uri: "http://localhost:19530"
  name: "rag_multimodal"
  collection_name: "collection_demo"
  namespace: "CaseDoneDemo"

# Retrieval Settings
retrieval:
  k: 2
  weights: [0.6, 0.4]
```

### Accessing Configuration

**Basic Usage**:
```python
from src.config import config

# Access configuration values
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

### Configuration Safety Features

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

Sensitive information like API keys should be stored in a `.env` file. The repository includes a `.env_sample` file as a template:

**Setup Steps**:
1. Copy the sample file: `cp .env_sample .env`
2. Edit `.env` with your actual API keys:

```
OPENAI_API_KEY=your_actual_openai_api_key_here
```

**Important**: Never commit your actual `.env` file to version control. The `.env_sample` file contains placeholder values and is safe to commit.

## Module Documentation

### AgenticRAG (`src/agent.py`)

The core agentic RAG implementation using LangGraph for intelligent document retrieval and generation.

**Key Features**:
- **Multiple Vector Store Support**: Configure multiple specialized vector stores
- **Intelligent Routing**: Agent automatically selects appropriate vector stores
- **Document Grading**: Evaluates document relevance before generation
- **Question Rewriting**: Improves queries for better retrieval
- **Dynamic Management**: Add/remove vector stores at runtime

**Usage**:
```python
from src.agent import AgenticRAG
from src.milvus_store import MilvusStore

# Single vector store (backward compatible)
milvus_store = MilvusStore()
rag_agent = AgenticRAG(vector_store=milvus_store)

# Multiple vector stores
vector_stores = [
    {
        'store': MilvusStore(collection_name="docs"),
        'name': 'search_documents',
        'description': 'Search technical documentation.',
        'k': 3
    },
    {
        'store': MilvusStore(collection_name="code"),
        'name': 'search_code',
        'description': 'Search code repositories.',
        'k': 2,
        'ranker_weights': [0.7, 0.3]
    }
]
rag_agent = AgenticRAG(vector_stores=vector_stores)

# Dynamic management
rag_agent.add_vector_store(
    store=new_store,
    name="search_api_docs",
    description="Search API documentation"
)
rag_agent.remove_vector_store("search_code")

# Get information about configured stores
info = rag_agent.get_vector_store_info()
```

### MilvusStore (`src/milvus_store.py`)

Vector database operations and management for storing and retrieving document embeddings.

**Key Features**:
- **Hybrid Search**: Combines dense and sparse retrieval with configurable weights
- **Namespace Support**: Partition data using namespaces
- **Database Management**: Create, drop, and manage collections
- **Configurable Logging**: Detailed logging with configurable levels
- **Connection Management**: Robust connection handling with error recovery

**Usage**:
```python
from src.milvus_store import MilvusStore

# Basic usage
milvus_store = MilvusStore()

# Custom configuration
milvus_store = MilvusStore(
    uri="http://localhost:19530",
    db_name="my_database",
    collection_name="my_collection",
    embed_model="text-embedding-3-large",
    namespace="my_namespace"
)

# Add documents
milvus_store.add_documents(documents)

# Create retriever with custom settings
retriever = milvus_store.as_retriever(
    k=5,
    namespace="specific_namespace",
    ranker_weights=[0.8, 0.2]
)

# Search operations
results = milvus_store.similarity_search("query", k=3)
results_with_scores = milvus_store.similarity_search_with_score("query", k=3)

# Database management
milvus_store.drop_collection("old_collection")
milvus_store.drop_all_collections(confirm=True)
```

### Document Processing (`src/index.py`)

Comprehensive document processing and indexing pipeline for various file formats.

**Key Features**:
- **Multi-format Support**: Process PDFs and other document formats
- **Multimodal Processing**: Handle text and images within documents
- **Configurable Chunking**: Flexible document chunking with custom parameters
- **Batch Processing**: Process entire directories efficiently
- **Custom Configuration**: Support for custom config files and parameters

**Usage**:
```python
from src.index import process_and_index_directory, get_chunker
from src.config import ConfigLoader

# Basic directory processing
process_and_index_directory("path/to/documents")

# Advanced processing with custom settings
process_and_index_directory(
    directory_path="path/to/documents",
    drop_existing=True,
    namespace="custom_namespace",
    file_extensions=[".pdf", ".docx"],
    config=custom_config
)

# Custom chunker configuration
chunker = get_chunker(
    tokenizer_model_id="sentence-transformers/all-MiniLM-L6-v2",
    max_tokens=1024,
    config=custom_config
)

# Custom configuration
custom_config = ConfigLoader("custom_config.yaml")
process_and_index_directory(
    "documents",
    config=custom_config
)
```

### Configuration Management (`src/config.py`)

Robust configuration system with safety features and flexible access patterns.

**Key Features**:
- **YAML-based Configuration**: Human-readable configuration files
- **Safety Controls**: Prevent accidental key creation with validation
- **Runtime Modification**: Change settings without restarting
- **Default Values**: Comprehensive fallback system
- **Custom Configs**: Support for multiple configuration instances

**Usage**:
```python
from src.config import config, ConfigLoader

# Basic access
model = config.get("model", "text_generation", default="gpt-4.1-mini")

# Runtime modification
config.set("database", "collection_name", "new_collection")

# Safety controls
config.set_allow_new_keys(False)  # Prevent new keys
config.set("new", "key", "value", force=True)  # Force creation

# Custom configuration
custom_config = ConfigLoader("custom.yaml", allow_new_keys=False)
value = custom_config.get("section", "key")

# Specialized configuration objects
milvus_args = config.get_milvus_connection_args()
pdf_options = config.get_pdf_pipeline_options()
```

### Web Interface (`src/ui.py`)

Gradio-based interactive web interface for easy interaction with the RAG system.

**Key Features**:
- **Chat Interface**: Intuitive conversation-based interaction
- **Message Processing**: Handles LangGraph message formats
- **Content Formatting**: Converts XML tags to Markdown for better display
- **Real-time Responses**: Streaming responses from the agent

**Usage**:
```python
from src.ui import GradioUI
from src.agent import AgenticRAG
from src.milvus_store import MilvusStore

# Create agent
milvus_store = MilvusStore()
rag_agent = AgenticRAG(vector_store=milvus_store)

# Launch web interface
ui = GradioUI(rag_agent, rag_agent.get_config())
ui.launch(
    share=True,
    server_name="0.0.0.0",
    server_port=7860
)
```

## Advanced Usage

### Custom Workflows

```python
from src.agent import AgenticRAG
from src.milvus_store import MilvusStore
from src.index import process_and_index_directory
from src.config import ConfigLoader

# 1. Setup custom configuration
custom_config = ConfigLoader("production.yaml")
custom_config.set("model", "text_generation", "gpt-4")
custom_config.set("retrieval", "k", 5)

# 2. Process documents with custom config
process_and_index_directory(
    "documents/technical",
    namespace="tech_docs",
    config=custom_config
)

process_and_index_directory(
    "documents/code",
    namespace="code_docs",
    config=custom_config
)

# 3. Create specialized vector stores
tech_store = MilvusStore(
    collection_name="tech_docs",
    namespace="tech_docs"
)
code_store = MilvusStore(
    collection_name="code_docs",
    namespace="code_docs"
)

# 4. Configure multi-store agent
vector_stores = [
    {
        'store': tech_store,
        'name': 'search_technical_docs',
        'description': 'Search technical documentation and manuals.',
        'k': 3
    },
    {
        'store': code_store,
        'name': 'search_code_examples',
        'description': 'Search code examples and implementation details.',
        'k': 2,
        'ranker_weights': [0.7, 0.3]
    }
]

# 5. Create and run agent
rag_agent = AgenticRAG(
    vector_stores=vector_stores,
    model_name="gpt-4",
    temperature=0.1
)

response = rag_agent.run("How do I implement authentication in the API?")
print(response)
```

### Logging Configuration

```python
import logging
from src.agent import AgenticRAG
from src.milvus_store import MilvusStore
from src.index import set_log_level

# Configure module-specific logging
AgenticRAG.configure_logging(logging.DEBUG, "logs/agent.log")
MilvusStore.set_log_level(logging.INFO, "logs/milvus.log")
set_log_level(logging.WARNING, "logs/index.log")

# Create agent with detailed logging
milvus_store = MilvusStore()
rag_agent = AgenticRAG(vector_store=milvus_store)
```

## Requirements

- Python 3.8+
- OpenAI API key
- Milvus vector database
- Dependencies listed in `pyproject.toml`

## Installation

```bash
# Install dependencies
poetry install

# Set up environment variables
cp .env_sample .env
# Edit .env with your actual API keys

# Start Milvus (using Docker)
docker run -d --name milvus -p 19530:19530 milvusdb/milvus:latest
```

## Contributing

The codebase follows these principles:

- **Modularity**: Each module has clear responsibilities
- **Configuration Safety**: Prevent accidental configuration errors
- **Comprehensive Logging**: Detailed logging for debugging and monitoring
- **Backward Compatibility**: Maintain compatibility with existing code
- **Type Safety**: Use type hints throughout the codebase