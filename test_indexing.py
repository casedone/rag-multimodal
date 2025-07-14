#!/usr/bin/env python3
"""
RAG Multimodal Ingestion Pipeline

This script processes documents from the specified directory and indexes them to a vector store.
It leverages the existing modules in the src directory for document processing and vector storage.

Usage:
    python main.py [--drop] [--namespace NAMESPACE]

Options:
    --drop          Drop existing collection if it exists
    --namespace     Namespace to use for the documents (default: from config)
"""

import os
import sys
import logging
import argparse
from pathlib import Path
from dotenv import load_dotenv

# Import from src modules
from src.index import process_and_index_directory, set_log_level
from src.config import Config

# Load environment variables
load_dotenv(".env")

# Configure logging
log_dir = Path("logs")
log_dir.mkdir(exist_ok=True)
set_log_level(logging.INFO, log_dir / "main.log")


def main():
    """Main entry point for the RAG ingestion pipeline."""
    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description="Process documents from a directory and index them to a vector store"
    )
    parser.add_argument(
        "--drop", 
        action="store_true", 
        help="Drop existing collection if it exists"
    )
    parser.add_argument(
        "--namespace", 
        help="Namespace to use for the documents"
    )
    args = parser.parse_args()
    
    # Set input directory
    doc_dir = Path("notebooks-scratch/doc_sources")
    
    # Verify OpenAI API key
    if not os.environ.get("OPENAI_API_KEY"):
        print("Error: OPENAI_API_KEY not found in environment variables")
        sys.exit(1)
    
    # Process and index documents
    print(f"Starting ingestion from {doc_dir}")
    try:
        process_and_index_directory(
            directory_path=doc_dir,
            drop_existing=args.drop,
            namespace=args.namespace
        )
        print("Ingestion completed successfully!")
    except Exception as e:
        print(f"Error during ingestion: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()