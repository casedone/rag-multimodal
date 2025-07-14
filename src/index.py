"""
Indexing module for RAG Multimodal application.
This module provides functionality to process files in a directory and index them to a vector store.

Usage:
    ```python
    from src.index import process_and_index_directory
    
    # Process and index all files in a directory
    process_and_index_directory("path/to/documents")
    
    # Process and index with custom configuration
    process_and_index_directory(
        directory_path="path/to/documents",
        drop_existing=True,
        namespace="custom_namespace"
    )
    ```
"""

import os
import copy
from pathlib import Path
from typing import List, Dict, Any, Optional, Union

# Import configuration
from src.config import Config

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
from langchain_docling import DoclingLoader
from langchain_docling.loader import ExportType
from langchain_core.documents import Document

# Import vector store libraries
from langchain_openai import OpenAIEmbeddings
from langchain_milvus import BM25BuiltInFunction, Milvus
from pymilvus import Collection, MilvusException, connections, db, utility
from transformers import AutoTokenizer


def initialize_vector_store(db_name: str) -> None:
    """
    Initialize the vector store database.
    
    Args:
        db_name: Name of the database to initialize
    """
    try:
        existing_databases = db.list_database()
        if db_name in existing_databases:
            print(f"Database '{db_name}' already exists.")
            # Use the database context
            db.using_database(db_name)
        else:
            print(f"Database '{db_name}' does not exist.")
            db.create_database(db_name)
            print(f"Database '{db_name}' created successfully.")
    except MilvusException as e:
        print(f"An error occurred: {e}")


def get_document_converter() -> DocumentConverter:
    """
    Create and configure a document converter.
    
    Returns:
        DocumentConverter: Configured document converter
    """
    pipeline_options = Config.get_pdf_pipeline_options()
    
    return DocumentConverter(
        format_options={
            InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)
        }
    )


def get_chunker() -> HybridChunker:
    """
    Create and configure a document chunker.
    
    Returns:
        HybridChunker: Configured document chunker
    """
    tokenizer = HuggingFaceTokenizer(
        tokenizer=AutoTokenizer.from_pretrained(Config.EMBED_MODEL_ID),
        max_tokens=Config.MAX_TOKENS,
    )
    
    class CustomMDSerializerProvider(ChunkingSerializerProvider):
        def get_serializer(self, doc):
            return ChunkingDocSerializer(
                doc=doc,
                table_serializer=MarkdownTableSerializer(),
                params=MarkdownParams(
                    image_mode=ImageRefMode.PLACEHOLDER,
                    image_placeholder="",
                    mark_annotations=True,
                    include_annotations=True
                )
            )
    
    return HybridChunker(
        tokenizer=tokenizer,
        serializer_provider=CustomMDSerializerProvider(),
    )


def get_vector_store(drop_old: bool = False) -> Milvus:
    """
    Create and configure a vector store.
    
    Args:
        drop_old: Whether to drop the existing collection if it exists
        
    Returns:
        Milvus: Configured vector store
    """
    # Connect to Milvus
    connections.connect(
        host=Config.URI.split("://")[1].split(":")[0], 
        port=int(Config.URI.split(":")[-1])
    )
    
    # Initialize the database
    initialize_vector_store(Config.DB_NAME)
    
    # Create embeddings model
    embeddings_model = OpenAIEmbeddings(
        model=Config.EMBED_MODEL,
        api_key=Config.OPENAI_API_KEY
    )
    
    # Create and return vector store
    return Milvus(
        embedding_function=embeddings_model,
        connection_args=Config.get_milvus_connection_args(),
        builtin_function=BM25BuiltInFunction(),
        vector_field=["dense", "sparse"],
        consistency_level="Strong",
        drop_old=drop_old,
        collection_name=Config.DB_COLLECTION_NAME,
        auto_id=True,
        partition_key_field="namespace"
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
        file_extensions: List of file extensions to process (defaults to ['.pdf', '.jpg', '.jpeg', '.png'])
    """
    directory_path = Path(directory_path) if isinstance(directory_path, str) else directory_path
    namespace = namespace or Config.NAMESPACE
    file_extensions = file_extensions or ['.pdf', '.jpg', '.jpeg', '.png']
    
    if not directory_path.exists():
        raise FileNotFoundError(f"Directory {directory_path} does not exist")
    
    # Get all files in the directory with specified extensions
    files = []
    for ext in file_extensions:
        files.extend(directory_path.glob(f"*{ext}"))
    
    if not files:
        print(f"No files with extensions {file_extensions} found in {directory_path}")
        return
    
    print(f"Found {len(files)} files to process")
    
    # Create document converter and chunker
    converter = get_document_converter()
    chunker = get_chunker()
    
    # Create vector store
    vector_store = get_vector_store(drop_old=drop_existing)
    
    # Process and index each file
    all_docs = []
    for file in files:
        if file.name == '.DS_Store':
            continue
        
        print(f"Processing {file}...")
        try:
            docs = process_file(file, converter, chunker, namespace)
            all_docs.extend(docs)
        except Exception as e:
            print(f"Error processing {file}: {e}")
    
    # Index all documents
    if all_docs:
        print(f"Indexing {len(all_docs)} documents...")
        try:
            ids = vector_store.add_documents(documents=all_docs)
            print(f"Successfully indexed {len(ids)} documents")
        except Exception as e:
            print(f"Error indexing documents: {e}")
    else:
        print("No documents to index")


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