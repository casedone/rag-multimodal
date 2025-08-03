"""
Agentic RAG implementation using LangGraph.

This module implements an agentic RAG (Retrieval Augmented Generation) system using LangGraph.
It supports multiple vector stores for document retrieval and uses a graph-based approach
to handle the RAG workflow with advanced features like document grading and question rewriting.

Usage:
    Single Vector Store (Backward Compatible):
    ```python
    from src.agent import AgenticRAG
    from src.milvus_store import MilvusStore
    
    # Initialize the vector store
    milvus_store = MilvusStore()
    
    # Create the agentic RAG system
    rag_agent = AgenticRAG(vector_store=milvus_store)
    
    # Run the RAG system with a query
    response = rag_agent.run("What is the main topic of the document?")
    print(response)
    ```
    
    Multiple Vector Stores:
    ```python
    from src.agent import AgenticRAG
    from src.milvus_store import MilvusStore
    
    # Initialize multiple vector stores
    docs_store = MilvusStore(collection_name="documents")
    code_store = MilvusStore(collection_name="code_base")
    
    # Configure vector stores with tool information
    vector_stores = [
        {
            'store': docs_store,
            'name': 'search_documents',
            'description': 'Search and retrieve information from document collection.'
        },
        {
            'store': code_store,
            'name': 'search_code',
            'description': 'Search and retrieve information from code repository.',
            'k': 3,  # Custom retrieval count
            'ranker_weights': [0.7, 0.3]  # Custom ranker weights
        }
    ]
    
    # Create the agentic RAG system with multiple stores
    rag_agent = AgenticRAG(vector_stores=vector_stores)
    
    # The agent will automatically choose the appropriate tool
    response = rag_agent.run("Show me the main function in the code")
    print(response)
    
    # Add more vector stores dynamically
    wiki_store = MilvusStore(collection_name="wikipedia")
    rag_agent.add_vector_store(
        store=wiki_store,
        name='search_wiki',
        description='Search Wikipedia articles for general knowledge.'
    )
    ```
"""

import logging
import os
import time
import uuid
from typing import Dict, List, Literal, Any, Optional, Union

# Import configuration
from src.config import config

# Import LangGraph components
from langgraph.graph import StateGraph, MessagesState, START, END
from langgraph.prebuilt import ToolNode
# Removed tools_condition import as we use custom routing
from langgraph.checkpoint.memory import InMemorySaver

# Import LangChain components
from langchain_core.messages import convert_to_messages, HumanMessage
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
    2. Retrieve relevant documents from multiple vector stores
    3. Grade document relevance
    4. Rewrite the question if needed
    5. Generate a final answer
    
    The system supports multiple vector stores, each converted to a retriever tool that
    the agent can access. Uses LangGraph for orchestrating the RAG workflow.
    """
    
    def __init__(
            self,
            vector_stores: Optional[List[Dict[str, Any]]] = None,
            vector_store: Optional[MilvusStore] = None,  # Backward compatibility
            model_name: str = None,
            api_key: str = None,
            temperature: float = 0,
            thread_id: Optional[str] = None,
            checkpointer: Optional[InMemorySaver] = None,
        ):
        """
        Initialize the AgenticRAG system.
        
        Args:
            vector_stores: List of vector store configurations. Each dict should contain:
                          - 'store': MilvusStore instance
                          - 'name': Tool name (string)
                          - 'description': Tool description (string)
                          - 'k': Optional retrieval count (defaults to config)
                          - 'ranker_weights': Optional ranker weights (defaults to config)
            vector_store: Single MilvusStore instance for backward compatibility
            model_name: Name of the model to use (defaults to config.get("model", "text_generation"))
            api_key: OpenAI API key (defaults to os.environ.get("OPENAI_API_KEY"))
            temperature: Temperature for the model (default: 0)
            thread_id: Optional thread ID for the conversation (default: None)
            checkpointer: Optional checkpointer for saving the conversation state (default: None)
        """
        self.model_name = model_name or config.get("model", "text_generation", default="gpt-4.1-mini")
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY", "")
        self.temperature = temperature
        self.checkpointer = checkpointer or InMemorySaver()
        
        # Generate unique thread_id if not provided
        if thread_id is None:
            self.thread_id = generate_thread_id()
        else:
            self.thread_id = thread_id
        
        # Initialize vector stores and create retriever tools
        self.vector_stores = []
        self.retriever_tools = []
        
        if vector_stores:
            # Use the new multiple vector stores approach
            for vs_config in vector_stores:
                if not isinstance(vs_config, dict):
                    raise ValueError("Each vector store configuration must be a dictionary")
                
                required_keys = ['store', 'name', 'description']
                missing_keys = [key for key in required_keys if key not in vs_config]
                if missing_keys:
                    raise ValueError(f"Vector store configuration missing required keys: {missing_keys}")
                
                store = vs_config['store']
                name = vs_config['name']
                description = vs_config['description']
                k = vs_config.get('k', config.get("retrieval", "k", default=2))
                ranker_weights = vs_config.get('ranker_weights', config.get("retrieval", "weights", default=[0.6, 0.4]))
                
                # Create retriever
                retriever = store.as_retriever(k=k, ranker_weights=ranker_weights)
                
                # Create retriever tool
                retriever_tool = create_retriever_tool(retriever, name, description)
                
                # Store the configuration
                self.vector_stores.append({
                    'store': store,
                    'name': name,
                    'description': description,
                    'retriever': retriever,
                    'tool': retriever_tool,
                    'k': k,
                    'ranker_weights': ranker_weights
                })
                self.retriever_tools.append(retriever_tool)
                
        elif vector_store:
            # Backward compatibility: single vector store
            retriever = vector_store.as_retriever(
                k=config.get("retrieval", "k", default=2),
                ranker_weights=config.get("retrieval", "weights", default=[0.6, 0.4])
            )
            
            retriever_tool = create_retriever_tool(
                retriever,
                "retrieve_documents",
                "Search and retrieve information from the document collection."
            )
            
            self.vector_stores.append({
                'store': vector_store,
                'name': 'retrieve_documents',
                'description': 'Search and retrieve information from the document collection.',
                'retriever': retriever,
                'tool': retriever_tool,
                'k': config.get("retrieval", "k", default=2),
                'ranker_weights': config.get("retrieval", "weights", default=[0.6, 0.4])
            })
            self.retriever_tools.append(retriever_tool)
            
        else:
            # Default: create a single MilvusStore
            default_store = MilvusStore()
            retriever = default_store.as_retriever(
                k=config.get("retrieval", "k", default=2),
                ranker_weights=config.get("retrieval", "weights", default=[0.6, 0.4])
            )
            
            retriever_tool = create_retriever_tool(
                retriever,
                "retrieve_documents",
                "Search and retrieve information from the document collection."
            )
            
            self.vector_stores.append({
                'store': default_store,
                'name': 'retrieve_documents',
                'description': 'Search and retrieve information from the document collection.',
                'retriever': retriever,
                'tool': retriever_tool,
                'k': config.get("retrieval", "k", default=2),
                'ranker_weights': config.get("retrieval", "weights", default=[0.6, 0.4])
            })
            self.retriever_tools.append(retriever_tool)
        
        # Maintain backward compatibility attributes
        self.vector_store = self.vector_stores[0]['store'] if self.vector_stores else None
        self.retriever = self.vector_stores[0]['retriever'] if self.vector_stores else None
        self.retriever_tool = self.retriever_tools[0] if self.retriever_tools else None
        
        # Initialize the LLM
        self.response_model = init_chat_model(self.model_name, temperature=self.temperature)
        self.grader_model = init_chat_model(self.model_name, temperature=0)
        
        # Create the graph
        self.graph = self._build_graph()
        
        logger.info(f"AgenticRAG initialized with model: {self.model_name} and {len(self.vector_stores)} vector store(s)")
    
    def add_vector_store(self, store: MilvusStore, name: str, description: str, 
                        k: Optional[int] = None, ranker_weights: Optional[List[float]] = None) -> None:
        """
        Add a new vector store to the agent.
        
        Args:
            store: MilvusStore instance
            name: Tool name for the retriever
            description: Tool description for the retriever
            k: Optional retrieval count (defaults to config)
            ranker_weights: Optional ranker weights (defaults to config)
        """
        k = k or config.get("retrieval", "k", default=2)
        ranker_weights = ranker_weights or config.get("retrieval", "weights", default=[0.6, 0.4])
        
        # Create retriever
        retriever = store.as_retriever(k=k, ranker_weights=ranker_weights)
        
        # Create retriever tool
        retriever_tool = create_retriever_tool(retriever, name, description)
        
        # Store the configuration
        vs_config = {
            'store': store,
            'name': name,
            'description': description,
            'retriever': retriever,
            'tool': retriever_tool,
            'k': k,
            'ranker_weights': ranker_weights
        }
        
        self.vector_stores.append(vs_config)
        self.retriever_tools.append(retriever_tool)
        
        # Rebuild the graph to include the new tool
        self.graph = self._build_graph()
        
        logger.info(f"Added vector store '{name}' to AgenticRAG")
    
    def remove_vector_store(self, name: str) -> bool:
        """
        Remove a vector store by name.
        
        Args:
            name: Name of the vector store to remove
            
        Returns:
            bool: True if removed successfully, False if not found
        """
        for i, vs_config in enumerate(self.vector_stores):
            if vs_config['name'] == name:
                # Remove from both lists
                removed_config = self.vector_stores.pop(i)
                self.retriever_tools.pop(i)
                
                # Update backward compatibility attributes if needed
                if self.vector_store == removed_config['store']:
                    self.vector_store = self.vector_stores[0]['store'] if self.vector_stores else None
                    self.retriever = self.vector_stores[0]['retriever'] if self.vector_stores else None
                    self.retriever_tool = self.retriever_tools[0] if self.retriever_tools else None
                
                # Rebuild the graph
                self.graph = self._build_graph()
                
                logger.info(f"Removed vector store '{name}' from AgenticRAG")
                return True
        
        logger.warning(f"Vector store '{name}' not found")
        return False
    
    def get_vector_store_info(self) -> List[Dict[str, Any]]:
        """
        Get information about all configured vector stores.
        
        Returns:
            List[Dict]: List of vector store information (excluding the actual store and tool objects)
        """
        return [{
            'name': vs['name'],
            'description': vs['description'],
            'k': vs['k'],
            'ranker_weights': vs['ranker_weights']
        } for vs in self.vector_stores]
    
    def _route_tools(self, state: MessagesState) -> str:
        """
        Custom routing function to route to the appropriate tool node or END.
        
        Use in the conditional_edge to route to the specific ToolNode if the last message
        has tool calls. Otherwise, route to the end.
        
        Args:
            state: Current state containing messages
            
        Returns:
            str: Node name to route to (tool name or END)
        """
        if isinstance(state, list):
            ai_message = state[-1]
        elif messages := state.get("messages", []):
            ai_message = messages[-1]
        else:
            raise ValueError(f"No messages found in input state to tool_edge: {state}")
            
        if hasattr(ai_message, "tool_calls") and len(ai_message.tool_calls) > 0:
            # Get the tool name from the first tool call
            tool_name = ai_message.tool_calls[0]["name"]
            
            # Verify that the tool name corresponds to one of our vector store tools
            valid_tool_names = [vs['name'] for vs in self.vector_stores]
            if tool_name in valid_tool_names:
                return tool_name
            else:
                logger.warning(f"Unknown tool name: {tool_name}. Available tools: {valid_tool_names}")
                return END
        
        return END
    
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
        
        # Add each retriever tool as an individual node
        retriever_node_names = []
        for vs_config in self.vector_stores:
            node_name = vs_config['name']
            retriever_node_names.append(node_name)
            workflow.add_node(node_name, ToolNode([vs_config['tool']]))
        
        workflow.add_node("rewrite_question", self._rewrite_question)
        workflow.add_node("generate_answer", self._generate_answer)
        
        # Add edges
        workflow.add_edge(START, "generate_query_or_respond")
        
        # Decide whether to retrieve - map each tool name to its corresponding node
        tools_mapping = {}
        for vs_config in self.vector_stores:
            tool_name = vs_config['name']
            tools_mapping[tool_name] = tool_name
        tools_mapping[END] = END
        
        workflow.add_conditional_edges(
            "generate_query_or_respond",
            self._route_tools,
            tools_mapping,
        )
        
        # Grade documents after retrieval from each retriever node
        for node_name in retriever_node_names:
            workflow.add_conditional_edges(
                node_name,
                self._grade_documents,
            )
        
        workflow.add_edge("generate_answer", END)
        workflow.add_edge("rewrite_question", "generate_query_or_respond")
        
        # Compile the graph
        graph = workflow.compile(checkpointer=self.checkpointer)

        output_file = "graph.png"
        graph.get_graph().draw_mermaid_png(output_file_path=output_file)
        return graph
    
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
            .bind_tools(self.retriever_tools)
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

        last_human_message = None
        for message in reversed(state["messages"]):
            if isinstance(message, HumanMessage):
                last_human_message = message
                break

        question = last_human_message.content
        
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
    
    def get_config(self) -> Dict[str, Any]:
        """
        Get the configuration dictionary for the agent.
        
        Returns:
            Dict[str, Any]: Configuration dictionary with thread_id
        """
        return {"configurable": {"thread_id": self.thread_id}}

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
        message = {"messages": [{"role": "user", "content": query}]}
        
        # Run the graph
        config = self.get_config()
        result = self.graph.invoke(message, config)
        
        # Extract the final response
        final_message = result["messages"][-1]
        response = final_message.content
        
        logger.info("Agentic RAG execution completed")
        
        return response