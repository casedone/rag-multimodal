"""
Agentic RAG implementation using LangGraph.

This module implements an agentic RAG (Retrieval Augmented Generation) system using LangGraph.
It integrates with the Milvus vector store for document retrieval and uses a graph-based approach
to handle the RAG workflow with advanced features like document grading and question rewriting.

Usage:
    ```python
    from src.agent import AgenticRAG
    from src.milvus_store import MilvusStore
    
    # Initialize the vector store
    milvus_store = MilvusStore()
    
    # Create the agentic RAG system
    rag_agent = AgenticRAG(milvus_store)
    
    # Run the RAG system with a query
    response = rag_agent.run("What is the main topic of the document?")
    print(response)
    ```
"""

import logging
import time
import uuid
from typing import Dict, List, Literal, Any, Optional, Union

# Import configuration
from src.config import Config

# Import LangGraph components
from langgraph.graph import StateGraph, MessagesState, START, END
from langgraph.prebuilt import ToolNode
from langgraph.prebuilt import tools_condition
from langgraph.checkpoint.memory import InMemorySaver

# Import LangChain components
from langchain_core.messages import convert_to_messages
from langchain.chat_models import init_chat_model
from langchain.tools.retriever import create_retriever_tool
from langchain_openai import OpenAIEmbeddings
from pydantic import BaseModel, Field

# Import Milvus store
from src.milvus_store import MilvusStore

# Configure logging
logger = logging.getLogger(__name__)

# Default log file path
DEFAULT_LOG_FILE = 'logs/agent.log'


def configure_logging(level=logging.INFO, log_file=None):
    """
    Configure logging for the agent module.
    
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
            import os
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
    
    logger.debug("Logging configured for agent module")


# Configure logging with default settings
configure_logging(log_file=DEFAULT_LOG_FILE)


def generate_thread_id() -> str:
    """
    Generate a unique thread ID using timestamp and UUID.
    
    Returns:
        str: Unique thread ID in format "{timestamp}_{random_uuid}"
    """
    timestamp = str(int(time.time() * 1000))  # milliseconds
    random_uuid = str(uuid.uuid4()).replace('-', '')  # 8 chars from UUID
    return f"{timestamp}_{random_uuid}"


class GradeDocuments(BaseModel):
    """Grade documents using a binary score for relevance check."""
    binary_score: str = Field(
        description="Relevance score: 'yes' if relevant, or 'no' if not relevant"
    )


class AgenticRAG:
    """
    Agentic RAG implementation using LangGraph.
    
    This class implements an agentic RAG system that uses a graph-based approach to:
    1. Generate a query or respond directly
    2. Retrieve relevant documents
    3. Grade document relevance
    4. Rewrite the question if needed
    5. Generate a final answer
    
    The system uses a Milvus vector store for document retrieval and LangGraph for
    orchestrating the RAG workflow.
    """
    
    def __init__(
            self,
            vector_store: Optional[MilvusStore] = None,
            model_name: str = None,
            api_key: str = None,
            temperature: float = 0,
            thread_id: Optional[str] = None,
            checkpointer: Optional[InMemorySaver] = None,
        ):
        """
        Initialize the AgenticRAG system.
        
        Args:
            vector_store: MilvusStore instance (if None, a new one will be created)
            model_name: Name of the model to use (defaults to Config.MODEL)
            api_key: OpenAI API key (defaults to Config.OPENAI_API_KEY)
            temperature: Temperature for the model (default: 0)
            thread_id: Optional thread ID for the conversation (default: None)
            checkpointer: Optional checkpointer for saving the conversation state (default: None)
        """
        self.model_name = model_name or Config.MODEL
        self.api_key = api_key or Config.OPENAI_API_KEY
        self.temperature = temperature
        self.checkpointer = checkpointer or InMemorySaver()
        
        # Generate unique thread_id if not provided
        if thread_id is None:
            self.thread_id = generate_thread_id()
        else:
            self.thread_id = thread_id
        
        # Initialize the vector store if not provided
        self.vector_store = vector_store or MilvusStore()
        
        # Create the retriever
        self.retriever = self.vector_store.as_retriever(
            k=Config.RETRIEVAL_K,
            ranker_weights=Config.RETRIEVAL_WEIGHTS
        )
        
        # Create the retriever tool
        self.retriever_tool = create_retriever_tool(
            self.retriever,
            "retrieve_documents",
            "Search and retrieve information from the document collection."
        )
        
        # Initialize the LLM
        self.response_model = init_chat_model(self.model_name, temperature=self.temperature)
        self.grader_model = init_chat_model(self.model_name, temperature=0)
        
        # Create the graph
        self.graph = self._build_graph()
        
        logger.info(f"AgenticRAG initialized with model: {self.model_name}")
    
    def _build_graph(self) -> StateGraph:
        """
        Build the LangGraph for the agentic RAG system.
        
        Returns:
            StateGraph: Compiled graph for the RAG workflow
        """
        # Create the workflow
        workflow = StateGraph(MessagesState)
        
        # Add nodes
        workflow.add_node("generate_query_or_respond", self._generate_query_or_respond)
        workflow.add_node("retrieve", ToolNode([self.retriever_tool]))
        workflow.add_node("rewrite_question", self._rewrite_question)
        workflow.add_node("generate_answer", self._generate_answer)
        
        # Add edges
        workflow.add_edge(START, "generate_query_or_respond")
        
        # Decide whether to retrieve
        workflow.add_conditional_edges(
            "generate_query_or_respond",
            tools_condition,
            {
                "tools": "retrieve",
                END: END,
            },
        )
        
        # Grade documents after retrieval
        workflow.add_conditional_edges(
            "retrieve",
            self._grade_documents,
        )
        
        workflow.add_edge("generate_answer", END)
        workflow.add_edge("rewrite_question", "generate_query_or_respond")
        
        # Compile the graph
        return workflow.compile(checkpointer=self.checkpointer)
    
    def _generate_query_or_respond(self, state: MessagesState) -> Dict:
        """
        Call the model to generate a response based on the current state.
        
        Given the input messages, it will decide to retrieve using the retriever tool,
        or respond directly to the user.
        
        Args:
            state: Current state containing messages
            
        Returns:
            Dict: Updated state with new messages
        """
        logger.debug("Generating query or response")
        response = (
            self.response_model
            .bind_tools([self.retriever_tool])
            .invoke(state["messages"])
        )
        return {"messages": [response]}
    
    def _grade_documents(
        self, 
        state: MessagesState
    ) -> Literal["generate_answer", "rewrite_question"]:
        """
        Determine whether the retrieved documents are relevant to the question.
        
        Args:
            state: Current state containing messages
            
        Returns:
            str: Next node to execute ("generate_answer" or "rewrite_question")
        """
        logger.debug("Grading retrieved documents")
        question = state["messages"][0].content
        context = state["messages"][-1].content
        
        grade_prompt = (
            "You are a grader assessing relevance of a retrieved document to a user question. \n "
            "Here is the retrieved document: \n\n {context} \n\n"
            "Here is the user question: {question} \n"
            "If the document contains keyword(s) or semantic meaning related to the user question, grade it as relevant. \n"
            "Give a binary score 'yes' or 'no' score to indicate whether the document is relevant to the question."
        )
        
        prompt = grade_prompt.format(question=question, context=context)
        response = (
            self.grader_model
            .with_structured_output(GradeDocuments)
            .invoke([{"role": "user", "content": prompt}])
        )
        score = response.binary_score
        
        logger.debug(f"Document relevance score: {score}")
        
        if score == "yes":
            return "generate_answer"
        else:
            return "rewrite_question"
    
    def _rewrite_question(self, state: MessagesState) -> Dict:
        """
        Rewrite the original user question to improve retrieval.
        
        Args:
            state: Current state containing messages
            
        Returns:
            Dict: Updated state with rewritten question
        """
        logger.debug("Rewriting question")
        messages = state["messages"]
        question = messages[0].content
        
        rewrite_prompt = (
            "Look at the input and try to reason about the underlying semantic intent / meaning.\n"
            "Here is the initial question:"
            "\n ------- \n"
            "{question}"
            "\n ------- \n"
            "Formulate an improved question:"
        )
        
        prompt = rewrite_prompt.format(question=question)
        response = self.response_model.invoke([{"role": "user", "content": prompt}])
        
        logger.debug(f"Original question: {question}")
        logger.debug(f"Rewritten question: {response.content}")
        
        return {"messages": [{"role": "user", "content": response.content}]}
    
    def _generate_answer(self, state: MessagesState) -> Dict:
        """
        Generate an answer based on the retrieved documents.
        
        Args:
            state: Current state containing messages
            
        Returns:
            Dict: Updated state with generated answer
        """
        logger.debug("Generating answer")
        question = state["messages"][0].content
        context = state["messages"][-1].content
        
        generate_prompt = (
            "You are an assistant for question-answering tasks. "
            "Use the following pieces of retrieved context to answer the question. "
            "If you don't know the answer, just say that you don't know. "
            "Use three sentences maximum and keep the answer concise.\n"
            "Question: {question} \n"
            "Context: {context}"
        )
        
        prompt = generate_prompt.format(question=question, context=context)
        response = self.response_model.invoke([{"role": "user", "content": prompt}])
        
        return {"messages": [response]}
    
    def run(self, query: str) -> str:
        """
        Run the agentic RAG system with a query.
        
        Args:
            query: User query
            
        Returns:
            str: Generated response
        """
        logger.info(f"Running agentic RAG with query: {query}")
        
        # Create initial state
        initial_state = {"messages": [{"role": "user", "content": query}]}
        
        # Run the graph
        config = {"configurable": {"thread_id": self.thread_id}}
        result = self.graph.invoke(initial_state, config)
        
        # Extract the final response
        final_message = result["messages"][-1]
        response = final_message.content
        
        logger.info("Agentic RAG execution completed")
        
        return response
    
    def update_thread_id(self, new_thread_id: Optional[str] = None) -> str:
        """
        Update the thread_id for the conversation.
        
        Args:
            new_thread_id: New thread ID to use. If None, generates a new unique thread_id.
            
        Returns:
            str: The updated thread_id
        """
        if new_thread_id is None:
            self.thread_id = generate_thread_id()
        else:
            self.thread_id = new_thread_id
            
        logger.info(f"Thread ID updated to: {self.thread_id}")
        return self.thread_id