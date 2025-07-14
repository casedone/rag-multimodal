"""
Milvus Vector Store module for RAG Multimodal application.
This module provides functionality to interact with Milvus vector database for storing and retrieving embeddings.

Usage:
    ```python
    from src.milvus_store import MilvusStore
    
    # Initialize the vector store
    milvus_store = MilvusStore()
    
    # Add documents to the vector store
    milvus_store.add_documents(documents)
    
    # Create a retriever
    retriever = milvus_store.as_retriever(k=3, namespace="custom_namespace")
    
    # Retrieve documents
    results = retriever.invoke("What is Docling?")
    ```
"""

import os
import logging
from typing import List, Dict, Any, Optional, Union

# Import configuration
from src.config import Config

# Import vector store libraries
from langchain_openai import OpenAIEmbeddings
from langchain_milvus import BM25BuiltInFunction, Milvus
from pymilvus import Collection, MilvusException, connections, db, utility
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever

# Configure logging
logger = logging.getLogger(__name__)

# Default log file path
DEFAULT_LOG_FILE = os.path.join('logs', 'milvus_store.log')


def configure_logging(level=logging.INFO, log_file=None):
    """
    Configure logging for the milvus_store module.
    
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
    
    logger.debug("Logging configured for milvus_store module")


# Configure logging with default settings
configure_logging(log_file=DEFAULT_LOG_FILE)


def drop_collection(collection_name: str, db_name: str) -> bool:
    """
    Drop a collection from the specified database.
    
    Args:
        collection_name: Name of the collection to drop
        db_name: Name of the database containing the collection
        
    Returns:
        bool: True if collection was successfully dropped, False otherwise
    """
    try:
        # Check if the database exists
        existing_databases = db.list_database()
        if db_name not in existing_databases:
            logger.warning(f"Database '{db_name}' does not exist.")
            return False
        
        # Switch to the specified database
        db.using_database(db_name)
        
        # Check if the collection exists
        collections = utility.list_collections()
        if collection_name not in collections:
            logger.warning(f"Collection '{collection_name}' does not exist in database '{db_name}'.")
            return False
        
        # Drop the collection
        collection = Collection(name=collection_name)
        collection.drop()
        logger.info(f"Collection '{collection_name}' has been dropped from database '{db_name}'.")
        return True
        
    except MilvusException as e:
        logger.error(f"Error dropping collection: {e}")
        return False


def drop_all_collections(db_name: str, confirm: bool = False) -> bool:
    """
    Drop all collections in a database.
    
    Args:
        db_name: Name of the database containing the collections
        confirm: Set to True to confirm the operation (defaults to False)
        
    Returns:
        bool: True if all collections were successfully dropped, False otherwise
    """
    if not confirm:
        logger.warning(f"WARNING: You are about to drop all collections in database '{db_name}'")
        logger.warning("This operation is irreversible. Set confirm=True to proceed.")
        return False
    
    try:
        # Check if the database exists
        existing_databases = db.list_database()
        if db_name not in existing_databases:
            logger.warning(f"Database '{db_name}' does not exist.")
            return False
        
        # Switch to the specified database
        db.using_database(db_name)
        
        # Get all collections in the database
        collections = utility.list_collections()
        logger.info(f"Found {len(collections)} collections in database '{db_name}'")
        
        # Drop each collection
        for collection_name in collections:
            logger.info(f"Dropping collection '{collection_name}'...")
            success = drop_collection(collection_name=collection_name, db_name=db_name)
            if not success:
                logger.error(f"Failed to drop collection '{collection_name}'")
                return False
            logger.info(f"Successfully dropped collection '{collection_name}'")
        
        return True
        
    except MilvusException as e:
        logger.error(f"Error dropping collections: {e}")
        return False


def drop_database(db_name: str, confirm: bool = False) -> bool:
    """
    Drop a database and all its collections.
    
    Args:
        db_name: Name of the database to drop
        confirm: Set to True to confirm the operation (defaults to False)
        
    Returns:
        bool: True if database was successfully dropped, False otherwise
    """
    if not confirm:
        logger.warning(f"WARNING: You are about to drop database '{db_name}'")
        logger.warning("This operation is irreversible. Set confirm=True to proceed.")
        return False
    
    try:
        # Check if the database exists
        existing_databases = db.list_database()
        if db_name not in existing_databases:
            logger.warning(f"Database '{db_name}' does not exist.")
            return False
        
        # First drop all collections in the database
        if not drop_all_collections(db_name, confirm=True):
            logger.error(f"Failed to drop all collections in database '{db_name}'")
            return False
        
        # Now drop the database
        db.drop_database(db_name)
        logger.info(f"Database '{db_name}' has been dropped.")
        return True
        
    except MilvusException as e:
        logger.error(f"Error dropping database: {e}")
        return False




class MilvusStore:
    """
    A class to manage interactions with Milvus vector database.
    
    This class provides functionality to interact with Milvus vector database for storing and retrieving embeddings.
    It includes comprehensive logging capabilities that can be configured using the set_log_level method.
    
    Logging Features:
    - Console logging (default at INFO level)
    - File logging (default to 'logs/milvus_store.log')
    - Configurable log levels (DEBUG, INFO, WARNING, ERROR, CRITICAL)
    - Formatted log messages with timestamps
    
    Example:
        ```python
        # Set custom debug level with custom file path
        MilvusStore.set_log_level(logging.DEBUG, 'custom/path/milvus.log')
        
        # Create store with default logging (INFO level)
        store = MilvusStore()
        ```
    """
    
    def __init__(
        self, 
        uri: str = None,
        db_name: str = None,
        collection_name: str = None,
        embed_model: str = None,
        api_key: str = None,
        drop_old: bool = False,
        namespace: str = None
    ):
        """
        Initialize the MilvusStore.
        
        Args:
            uri: URI for Milvus connection (defaults to Config.URI)
            db_name: Name of the database (defaults to Config.DB_NAME)
            collection_name: Name of the collection (defaults to Config.DB_COLLECTION_NAME)
            embed_model: Embedding model to use (defaults to Config.EMBED_MODEL)
            api_key: OpenAI API key (defaults to Config.OPENAI_API_KEY)
            drop_old: Whether to drop the existing collection if it exists
            namespace: Default namespace to use for documents (defaults to Config.NAMESPACE)
        """
        self.uri = uri or Config.URI
        self.db_name = db_name or Config.DB_NAME
        self.collection_name = collection_name or Config.DB_COLLECTION_NAME
        self.embed_model = embed_model or Config.EMBED_MODEL
        self.api_key = api_key or Config.OPENAI_API_KEY
        self.namespace = namespace or Config.NAMESPACE
        
        # Connect to Milvus
        self._connect_to_milvus()
        
        # Initialize the database
        self._initialize_vector_store(drop_old=drop_old)
        
        # Create embeddings model
        self.embeddings_model = OpenAIEmbeddings(
            model=self.embed_model,
            api_key=self.api_key
        )
        
        # Create vector store
        self.vector_store = self._create_vector_store(drop_old=drop_old)
    
    def _connect_to_milvus(self) -> None:
        """
        Connect to Milvus server.
        """
        host = self.uri.split("://")[1].split(":")[0]
        port = int(self.uri.split(":")[-1])
        connections.connect(host=host, port=port)
    
    def _initialize_vector_store(self, drop_old: bool = False) -> None:
        """
        Initialize the vector store database.
        
        Args:
            drop_old: Whether to drop the existing database if it exists
        """
        try:
            existing_databases = db.list_database()
            if self.db_name in existing_databases:
                logger.info(f"Database '{self.db_name}' already exists.")
                
                # Use the database context
                db.using_database(self.db_name)
                
                if drop_old:
                    # Drop the collection if it exists
                    collections = utility.list_collections()
                    if self.collection_name in collections:
                        drop_collection(self.collection_name, self.db_name)
            else:
                logger.info(f"Database '{self.db_name}' does not exist.")
                db.create_database(self.db_name)
                logger.info(f"Database '{self.db_name}' created successfully.")
        except MilvusException as e:
            logger.error(f"An error occurred: {e}")
    
    def _create_vector_store(self, drop_old: bool = False) -> Milvus:
        """
        Create and configure a vector store.
        
        Args:
            drop_old: Whether to drop the existing collection if it exists
            
        Returns:
            Milvus: Configured vector store
        """
        connection_args = {
            "uri": self.uri,
            "db_name": self.db_name
        }
        
        # Create and return vector store
        return Milvus(
            embedding_function=self.embeddings_model,
            connection_args=connection_args,
            builtin_function=BM25BuiltInFunction(),
            vector_field=["dense", "sparse"],
            consistency_level="Strong",
            drop_old=drop_old,
            collection_name=self.collection_name,
            auto_id=True,
            partition_key_field="namespace"
        )
    
    def add_documents(self, documents: List[Document]) -> List[str]:
        """
        Add documents to the vector store.
        
        Args:
            documents: List of documents to add
            
        Returns:
            List[str]: List of document IDs
        """
        if not documents:
            logger.warning("No documents to add")
            return []
        
        try:
            ids = self.vector_store.add_documents(documents=documents)
            logger.info(f"Successfully added {len(ids)} documents")
            return ids
        except Exception as e:
            logger.error(f"Error adding documents: {e}")
            return []
    
    def as_retriever(
        self, 
        k: int = 4, 
        namespace: str = None,
        ranker_type: str = "weighted",
        ranker_weights: List[float] = None
    ) -> BaseRetriever:
        """
        Create a retriever from the vector store.
        
        Args:
            k: Number of documents to retrieve
            namespace: Namespace to filter by (defaults to self.namespace)
            ranker_type: Type of ranker to use (default: "weighted")
            ranker_weights: Weights for the ranker (default: [0.6, 0.4])
            
        Returns:
            BaseRetriever: Configured retriever
        """
        namespace = namespace or self.namespace
        ranker_weights = ranker_weights or [0.6, 0.4]
        
        search_kwargs = {
            "k": k
        }
        
        if namespace:
            search_kwargs["expr"] = f'namespace == "{namespace}"'
        
        return self.vector_store.as_retriever(
            search_kwargs=search_kwargs,
            ranker_type=ranker_type,
            ranker_params={"weights": ranker_weights}
        )
    
    def similarity_search(
        self, 
        query: str, 
        k: int = 4, 
        namespace: str = None
    ) -> List[Document]:
        """
        Perform a similarity search.
        
        Args:
            query: Query string
            k: Number of documents to retrieve
            namespace: Namespace to filter by (defaults to self.namespace)
            
        Returns:
            List[Document]: List of similar documents
        """
        namespace = namespace or self.namespace
        
        filter_expr = None
        if namespace:
            filter_expr = f'namespace == "{namespace}"'
        
        return self.vector_store.similarity_search(
            query,
            k=k,
            expr=filter_expr
        )
    
    def similarity_search_with_score(
        self, 
        query: str, 
        k: int = 4, 
        namespace: str = None
    ) -> List[tuple]:
        """
        Perform a similarity search with scores.
        
        Args:
            query: Query string
            k: Number of documents to retrieve
            namespace: Namespace to filter by (defaults to self.namespace)
            
        Returns:
            List[tuple]: List of (document, score) tuples
        """
        namespace = namespace or self.namespace
        
        filter_expr = None
        if namespace:
            filter_expr = f'namespace == "{namespace}"'
        
        return self.vector_store.similarity_search_with_score(
            query,
            k=k,
            expr=filter_expr
        )
    
    @staticmethod
    def set_log_level(level=logging.INFO, log_file=None):
        """
        Set the logging level for the MilvusStore module.
        
        Args:
            level: Logging level (e.g., logging.DEBUG, logging.INFO, logging.WARNING)
            log_file: Optional path to log file
            
        Example:
            ```python
            # Set to debug level with file logging
            MilvusStore.set_log_level(logging.DEBUG, 'logs/milvus.log')
            
            # Set to warning level with console only
            MilvusStore.set_log_level(logging.WARNING)
            ```
        """
        configure_logging(level=level, log_file=log_file)
        logger.info(f"Log level set to: {logging.getLevelName(level)}")
    
    def drop_collection(self, collection_name: str = None, db_name: str = None) -> bool:
        """
        Drop a collection from the specified database.
        
        Args:
            collection_name: Name of the collection to drop (defaults to self.collection_name)
            db_name: Name of the database containing the collection (defaults to self.db_name)
            
        Returns:
            bool: True if collection was successfully dropped, False otherwise
        """
        collection_name = collection_name or self.collection_name
        db_name = db_name or self.db_name
        
        return drop_collection(collection_name, db_name)
    
    def drop_all_collections(self, db_name: str = None, confirm: bool = False) -> bool:
        """
        Drop all collections in a database.
        
        Args:
            db_name: Name of the database containing the collections (defaults to self.db_name)
            confirm: Set to True to confirm the operation (defaults to False)
            
        Returns:
            bool: True if all collections were successfully dropped, False otherwise
        """
        db_name = db_name or self.db_name
        
        return drop_all_collections(db_name, confirm)
    
    def drop_database(self, db_name: str = None, confirm: bool = False) -> bool:
        """
        Drop a database and all its collections.
        
        Args:
            db_name: Name of the database to drop (defaults to self.db_name)
            confirm: Set to True to confirm the operation (defaults to False)
            
        Returns:
            bool: True if database was successfully dropped, False otherwise
        """
        db_name = db_name or self.db_name
        
        return drop_database(db_name, confirm)
    

