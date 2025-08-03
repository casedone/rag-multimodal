"""
Indexing module for RAG Multimodal application.
This module provides functionality to process files in a directory and index them to a vector store.

The module includes comprehensive logging capabilities that can be configured using the set_log_level function.

Usage:
    ```python
    from src.index import process_and_index_directory, set_log_level
    import logging
    
    # Configure custom logging (optional)
    set_log_level(logging.DEBUG, 'custom/path/index.log')
    
    # Process and index all files in a directory
    process_and_index_directory("path/to/documents")
    
    # Process and index with custom configuration
    process_and_index_directory(
        directory_path="path/to/documents",
        drop_existing=True,
        namespace="custom_namespace"
    )
    ```
    
Logging Features:
    - Console logging (default at INFO level)
    - File logging (default to 'logs/index.log')
    - Configurable log levels (DEBUG, INFO, WARNING, ERROR, CRITICAL)
    - Formatted log messages with timestamps
"""

import os
import copy
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional, Union

# Import configuration
from src.config import Config, ConfigLoader

# Configure logging
logger = logging.getLogger(__name__)

# Default log file path
DEFAULT_LOG_FILE = os.path.join('logs', 'index.log')

# Import document processing libraries
from docling_core.transforms.chunker.tokenizer.huggingface import HuggingFaceTokenizer
from docling_core.transforms.chunker.hierarchical_chunker import (
    ChunkingDocSerializer,
    ChunkingSerializerProvider,
)
from docling_core.transforms.serializer.markdown import MarkdownTableSerializer, MarkdownParams
from docling_core.types.doc import ImageRefMode
from docling.datamodel.base_models import InputFormat
from docling.document_converter import DocumentConverter, PdfFormatOption
from docling.chunking import HybridChunker
from docling.datamodel.pipeline_options import (
            PdfPipelineOptions,
            PictureDescriptionApiOptions
        )
from langchain_docling import DoclingLoader
from langchain_docling.loader import ExportType
from langchain_core.documents import Document
from transformers import AutoTokenizer

# Import our Milvus store module
from src.milvus_store import MilvusStore


def set_log_level(level=logging.INFO, log_file=None):
    """
    Set the logging level for the index module.
    
    Args:
        level: Logging level (e.g., logging.DEBUG, logging.INFO, logging.WARNING)
        log_file: Optional path to log file
        
    Example:
        ```python
        # Set to debug level with file logging
        from src.index import set_log_level
        import logging
        
        set_log_level(logging.DEBUG, 'logs/index.log')
        ```
    """
    setup_logging(level=level, log_file=log_file)
    logger.info(f"Log level set to: {logging.getLevelName(level)}")


def setup_logging(level=logging.INFO, log_file=None):
    """
    Configure logging for the index module.
    
    Args:
        level: Logging level (default: logging.INFO)
        log_file: Path to log file (default: None, logs to console only)
    """
    # Create a formatter
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    
    # Configure the logger
    logger.setLevel(level)
    
    # Remove existing handlers to avoid duplicates
    for handler in logger.handlers[:]:  
        logger.removeHandler(handler)
    
    # Create console handler and set level
    console_handler = logging.StreamHandler()
    console_handler.setLevel(level)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # Add file handler if log_file is specified
    if log_file:
        try:
            # Create directory for log file if it doesn't exist
            log_dir = os.path.dirname(log_file)
            if log_dir and not os.path.exists(log_dir):
                os.makedirs(log_dir)
                
            # Create file handler and set level
            file_handler = logging.FileHandler(log_file)
            file_handler.setLevel(level)
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)
            logger.info(f"Logging to file: {log_file}")
        except Exception as e:
            logger.error(f"Failed to set up file logging to {log_file}: {e}")
    
    logger.debug("Logging configured for index module")


# Configure logging with default settings
setup_logging(log_file=DEFAULT_LOG_FILE)


def get_document_converter(pdf_pipeline_options: Optional[PdfPipelineOptions] = None) -> DocumentConverter:
    """
    Create and configure a document converter.
    
    Returns:
        DocumentConverter: Configured document converter
    """
    pdf_pipeline_options = pdf_pipeline_options or Config.get_pdf_pipeline_options()
    
    return DocumentConverter(
        format_options={
            InputFormat.PDF: PdfFormatOption(pipeline_options=pdf_pipeline_options)
        }
    )


def get_chunker(
    embed_model_id: Optional[str] = None,
    max_tokens: Optional[int] = None,
    image_mode: Optional[ImageRefMode] = None,
    image_placeholder: Optional[str] = None,
    mark_annotations: Optional[bool] = None,
    include_annotations: Optional[bool] = None,
    config: Optional['ConfigLoader'] = None
) -> HybridChunker:
    """
    Create and configure a document chunker with optional configuration overrides.
    
    Args:
        embed_model_id: Model ID for tokenizer (defaults to Config.EMBED_MODEL_ID)
        max_tokens: Maximum tokens per chunk (defaults to Config.MAX_TOKENS)
        image_mode: Image reference mode (defaults to ImageRefMode.PLACEHOLDER)
        image_placeholder: Placeholder text for images (defaults to "")
        mark_annotations: Whether to mark annotations (defaults to True)
        include_annotations: Whether to include annotations (defaults to True)
        config: Optional ConfigLoader instance to use instead of global Config
    
    Returns:
        HybridChunker: Configured document chunker
    """
    # Use provided values or fall back to config defaults
    if config:
        # Use custom config
        default_model_id = config.get('model', 'tokenizer', default='sentence-transformers/all-MiniLM-L6-v2')
        default_max_tokens = config.get('document', 'max_tokens', default=512)
    else:
        # Use global Config
        default_model_id = Config.EMBED_MODEL_ID
        default_max_tokens = Config.MAX_TOKENS
    
    model_id = embed_model_id or default_model_id
    tokens = max_tokens or default_max_tokens
    img_mode = image_mode if image_mode is not None else ImageRefMode.PLACEHOLDER
    img_placeholder = image_placeholder if image_placeholder is not None else ""
    mark_annot = mark_annotations if mark_annotations is not None else True
    include_annot = include_annotations if include_annotations is not None else True
    
    tokenizer = HuggingFaceTokenizer(
        tokenizer=AutoTokenizer.from_pretrained(model_id),
        max_tokens=tokens,
    )
    
    class CustomMDSerializerProvider(ChunkingSerializerProvider):
        def get_serializer(self, doc):
            return ChunkingDocSerializer(
                doc=doc,
                table_serializer=MarkdownTableSerializer(),
                params=MarkdownParams(
                    image_mode=img_mode,
                    image_placeholder=img_placeholder,
                    mark_annotations=mark_annot,
                    include_annotations=include_annot
                )
            )
    
    return HybridChunker(
        tokenizer=tokenizer,
        serializer_provider=CustomMDSerializerProvider(),
    )


def process_file(file_path: Union[str, Path], converter: DocumentConverter, chunker: HybridChunker, namespace: str) -> List[Document]:
    """
    Process a single file and prepare documents for indexing.
    
    Args:
        file_path: Path to the file to process
        converter: Document converter to use
        chunker: Document chunker to use
        namespace: Namespace to use for the documents
        
    Returns:
        List[Document]: List of processed documents
    """
    file_path = Path(file_path) if isinstance(file_path, str) else file_path
    
    # Create document loader
    loader = DoclingLoader(
        file_path=file_path,
        converter=converter,
        chunker=chunker,
        export_type=ExportType.DOC_CHUNKS
    )
    
    # Load and process documents
    docs = loader.load()
    
    # Prepare documents for indexing
    processed_docs = []
    for doc in docs:
        metadata = doc.metadata
        _metadata = dict()
        _metadata["source"] = str(metadata["source"])
        _metadata["page_no"] = metadata["dl_meta"]['doc_items'][0]['prov'][0]['page_no']
        _metadata["namespace"] = namespace
        
        processed_doc = Document(
            page_content=doc.page_content,
            metadata=_metadata
        )
        processed_docs.append(processed_doc)
    
    return processed_docs


def process_and_index_directory(
    directory_path: Union[str, Path],
    drop_existing: bool = False,
    namespace: str = None,
    file_extensions: List[str] = None
) -> None:
    """
    Process all files in a directory and index them to a vector store.
    
    Args:
        directory_path: Path to the directory containing files to process
        drop_existing: Whether to drop the existing collection if it exists
        namespace: Namespace to use for the documents (defaults to Config.NAMESPACE)
        file_extensions: List of file extensions to process (defaults to Config.SUPPORTED_FILE_TYPES)
    """
    directory_path = Path(directory_path) if isinstance(directory_path, str) else directory_path
    namespace = namespace or Config.NAMESPACE
    file_extensions = file_extensions or Config.SUPPORTED_FILE_TYPES
    
    if not directory_path.exists():
        raise FileNotFoundError(f"Directory {directory_path} does not exist")
    
    # Get all files in the directory with specified extensions
    files = []
    for ext in file_extensions:
        files.extend(directory_path.glob(f"*{ext}"))
    
    if not files:
        logger.warning(f"No files with extensions {file_extensions} found in {directory_path}")
        return
    
    logger.info(f"Found {len(files)} files to process")
    
    # Create document converter and chunker
    converter = get_document_converter()
    chunker = get_chunker()
    
    # Create vector store
    milvus_store = MilvusStore(drop_old=drop_existing, namespace=namespace)
    
    # Process and index each file
    all_docs = []
    for file in files:
        if file.name == '.DS_Store':
            continue
        
        logger.info(f"Processing {file}...")
        try:
            docs = process_file(file, converter, chunker, namespace)
            all_docs.extend(docs)
        except Exception as e:
            logger.error(f"Error processing {file}: {e}")
    
    # Index all documents
    if all_docs:
        logger.info(f"Indexing {len(all_docs)} documents...")
        ids = milvus_store.add_documents(documents=all_docs)
        if not ids:
            logger.error("Failed to index documents")
    else:
        logger.warning("No documents to index")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Process and index files in a directory")
    parser.add_argument("directory", help="Directory containing files to process")
    parser.add_argument("--drop", action="store_true", help="Drop existing collection if it exists")
    parser.add_argument("--namespace", help="Namespace to use for the documents")
    
    args = parser.parse_args()
    
    process_and_index_directory(
        directory_path=args.directory,
        drop_existing=args.drop,
        namespace=args.namespace
    )