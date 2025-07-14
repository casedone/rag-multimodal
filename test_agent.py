"""
Test module for the AgenticRAG implementation.

This script demonstrates how to use the AgenticRAG class with sample queries.
It initializes the Milvus vector store and the agentic RAG system, then runs
a series of test queries to showcase the system's capabilities.

Usage:
    ```bash
    # Run the test script
    python test_agent.py
    
    # Run with a specific query
    python test_agent.py --query "What is the main topic of the document?"
    ```
"""

import os
import sys
import logging
import argparse
from pathlib import Path

# Import the AgenticRAG and MilvusStore classes
from src.agent import AgenticRAG
from src.milvus_store import MilvusStore
from src.config import Config

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('logs/test_agent.log', mode='a')
    ]
)

logger = logging.getLogger(__name__)


def ensure_documents_exist(milvus_store):
    """
    Check if documents exist in the vector store and process them if not.
    
    Args:
        milvus_store: MilvusStore instance
    """
    try:
        # Try to retrieve a document to check if the collection has data
        test_retriever = milvus_store.as_retriever(k=1)
        test_results = test_retriever.invoke("test")
        
        if not test_results:
            logger.info("No documents found in the vector store. Processing sample documents...")
            # Import the document processing function
            from src.index import process_and_index_directory
            
            # Process and index documents from the configured directory
            doc_dir = Config.DOC_DIR
            if not os.path.exists(doc_dir):
                logger.warning(f"Document directory {doc_dir} does not exist. Creating it...")
                os.makedirs(doc_dir)
                logger.info(f"Please add documents to {doc_dir} and run this script again.")
                sys.exit(1)
                
            process_and_index_directory(doc_dir, drop_existing=True)
        else:
            logger.info("Documents found in the vector store.")
    except Exception as e:
        logger.error(f"Error checking for documents: {e}")
        raise


def run_test_queries(rag_agent, queries=None):
    """
    Run a series of test queries through the RAG agent.
    
    Args:
        rag_agent: AgenticRAG instance
        queries: List of queries to run (if None, default queries will be used)
    """
    if queries is None:
        queries = [
            "What is DocLing good for?"
            # "What is the main topic of the document?",
            # "Summarize the key points in the document.",
            # "What are the most important concepts discussed?",
            # "How does this information relate to machine learning?",
            # "What are the practical applications mentioned in the document?"
        ]
    
    logger.info(f"Running {len(queries)} test queries...")
    
    for i, query in enumerate(queries, 1):
        logger.info(f"\nQuery {i}: {query}")
        try:
            response = rag_agent.run(query)
            logger.info(f"Response: {response}")
        except Exception as e:
            logger.error(f"Error processing query: {e}")


def main():
    """Main function to run the test script."""
    parser = argparse.ArgumentParser(description="Test the AgenticRAG implementation.")
    parser.add_argument("--query", type=str, help="A specific query to test")
    args = parser.parse_args()
    
    logger.info("Initializing Milvus vector store...")
    milvus_store = MilvusStore()
    
    # Ensure documents exist in the vector store
    ensure_documents_exist(milvus_store)
    
    logger.info("Initializing AgenticRAG system...")
    rag_agent = AgenticRAG(vector_store=milvus_store)
    
    if args.query:
        # Run a single query if provided
        logger.info(f"Running query: {args.query}")
        try:
            response = rag_agent.run(args.query)
            logger.info(f"Response: {response}")
        except Exception as e:
            logger.error(f"Error processing query: {e}")
    else:
        # Run the default test queries
        run_test_queries(rag_agent)
    
    logger.info("Test completed.")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logger.error(f"An error occurred: {e}", exc_info=True)
        sys.exit(1)
