import gradio as gr
import re
from langchain_core.messages.system import SystemMessage
from langchain_core.messages.human import HumanMessage
from langchain_core.messages.ai import AIMessage
from langchain_core.messages.tool import ToolMessage
from typing import Any, Dict, List, Tuple

def preprocess_content(content):
    """Replace XML-style tags with Markdown headings, handling nested tags"""
    if not isinstance(content, str):
        return content
    
    # First pass: identify all tags and their positions
    tag_positions = []
    for match in re.finditer(r'</?(\w+)>', content):
        is_closing = match.group(0).startswith('</')
        tag_name = match.group(1)
        tag_positions.append((match.start(), match.end(), tag_name, is_closing))
    
    # If no tags found, return the original content
    if not tag_positions:
        return content
    
    # Sort by position to process in order
    tag_positions.sort()
    
    # Second pass: replace tags with appropriate heading levels
    result = []
    last_pos = 0
    nesting_level = 0
    tag_stack = []
    
    for start, end, tag_name, is_closing in tag_positions:
        # Add text before this tag
        result.append(content[last_pos:start])
        
        if is_closing:
            # Handle closing tag
            if tag_stack and tag_stack[-1] == tag_name:
                tag_stack.pop()
                nesting_level = max(0, nesting_level - 1)
            # Skip adding the closing tag
        else:
            # Handle opening tag
            heading_level = min(6, 2 + nesting_level)  # Start at ## (h2), max at ###### (h6)
            result.append('#' * heading_level + ' ' + tag_name.capitalize())
            tag_stack.append(tag_name)
            nesting_level += 1
        
        last_pos = end
    
    # Add remaining text
    result.append(content[last_pos:])
    
    return ''.join(result)

def get_messages_from_langgraph_state(state):
    """Extract ChatMessage objects from agent's state and transform to messages suitable for Gradio"""

    try:
        messages = state.values['messages']
    except (KeyError, AttributeError):
        messages = []
    
    return messages


def convert_messages_from_langchain_to_gradio(messages):
    """Extract ChatMessage objects from agent's state and transform to messages suitable for Gradio"""
    
    i = 0
    while i < len(messages):
        message = messages[i]
        
        if isinstance(message, SystemMessage):
            # System messages are typically not shown in the UI
            i += 1
            continue
            
        elif isinstance(message, HumanMessage):
            # Process human message content to replace XML tags with Markdown headings
            processed_content = preprocess_content(message.content)
            yield gr.ChatMessage(role="user", content=processed_content)
            i += 1
            
        elif isinstance(message, AIMessage):
            # For AI messages, yield the content
            ai_content = preprocess_content(message.content)
            
            # Check if this AI message has tool calls
            has_tool_calls = hasattr(message, 'tool_calls') and message.tool_calls
            
            # Yield the AI message content
            yield gr.ChatMessage(role="assistant", content=ai_content)
            
            # Check if this AI message has tool calls and is followed by a ToolMessage
            if has_tool_calls and i + 1 < len(messages) and isinstance(messages[i + 1], ToolMessage):
                tool_message = messages[i + 1]
                
                # Find the matching tool call
                for tool_call in message.tool_calls:
                    if tool_call['id'] == tool_message.tool_call_id:
                        # Generate a unique ID for this tool call
                        parent_id = f"call_{tool_call['id']}"
                        
                        # Format the tool call content
                        # Handle arguments based on type, similar to gradio_ui.py
                        if 'args' in tool_call and isinstance(tool_call['args'], dict):
                            content = str(tool_call['args'].get("answer", str(tool_call['args'])))
                        else:
                            content = preprocess_content(tool_message.content)
                        
                        # Special handling for code-related tools
                        used_code = tool_call['name'] == "python_interpreter" if 'name' in tool_call else False
                        if used_code:
                            # Clean up the content for code display
                            import re
                            content = re.sub(r"```.*?\n", "", content)  # Remove existing code blocks
                            content = re.sub(r"\s*<end_code>\s*", "", content)  # Remove end_code tags
                            content = content.strip()
                            if not content.startswith("```python"):
                                content = f"```python\n{content}\n```"
                        
                        # Create a parent message for the tool call with pending status
                        parent_message_tool = gr.ChatMessage(
                            role="assistant",
                            content=content,
                            metadata={
                                "title": f"🛠️ Used tool {tool_call['name']}",
                                "id": parent_id,
                                "status": "pending",
                            }
                        )
                        yield parent_message_tool
                        
                        # If there are observations in the tool message, display them as a child
                        if hasattr(tool_message, 'observations') and tool_message.observations:
                            log_content = str(tool_message.observations).strip()
                            if log_content:
                                # Clean up the log content
                                import re
                                log_content = re.sub(r"^Execution logs:\s*", "", log_content)
                                yield gr.ChatMessage(
                                    role="assistant",
                                    content=f"```bash\n{log_content}\n```",
                                    metadata={
                                        "title": "📝 Execution Logs",
                                        "parent_id": parent_id,
                                        "status": "done"
                                    }
                                )
                        
                        # If there's an error in the tool message, display it as a child
                        if hasattr(tool_message, 'error') and tool_message.error is not None:
                            yield gr.ChatMessage(
                                role="assistant",
                                content=str(tool_message.error),
                                metadata={
                                    "title": "💥 Error",
                                    "parent_id": parent_id,
                                    "status": "done"
                                }
                            )
                        
                        # Update parent message status to done
                        parent_message_tool.metadata["status"] = "done"
                        
                        # Add a step footnote with token counts and duration if available
                        step_footnote = ""
                        if hasattr(message, 'input_token_count') and hasattr(message, 'output_token_count'):
                            token_str = f" | Input-tokens:{message.input_token_count:,} | Output-tokens:{message.output_token_count:,}"
                            step_footnote += token_str
                        
                        if hasattr(message, 'duration'):
                            step_duration = f" | Duration: {round(float(message.duration), 2)}" if message.duration else None
                            if step_duration:
                                step_footnote += step_duration
                        
                        if step_footnote:
                            step_footnote = f"""<span style="color: #bbbbc2; font-size: 12px;">{step_footnote}</span> """
                            yield gr.ChatMessage(role="assistant", content=f"{step_footnote}")
                            
                            # Add a separator line
                            yield gr.ChatMessage(role="assistant", content="-----", metadata={"status": "done"})
                        
                        break
                
                # Skip the tool message as we've already processed it
                i += 2
            else:
                i += 1
        
        elif isinstance(message, ToolMessage):
            # If we encounter a ToolMessage without a preceding AIMessage with tool_calls
            # (this shouldn't normally happen, but just in case)
            processed_content = preprocess_content(message.content)
            yield gr.ChatMessage(
                role="assistant",
                content=processed_content,
                metadata={
                    "title": f"🛠️ Tool Result: {message.name}",
                    "status": "done"
                }
            )
            i += 1
        
        else:
            # Skip other message types
            i += 1
    
    return

class GradioUI:
    """A Gradio-based user interface for interacting with LangGraph agents.
    
    This class provides a chat interface that connects to a LangGraph agent,
    allowing users to interact with the agent through a web UI.
    """

    def __init__(self, agent: Any, config: Dict[str, Any]) -> None:
        """Initialize the Gradio UI.
        
        Args:
            agent: The LangGraph agent to interact with
            config: Configuration dictionary for the agent
        """
        self.agent = agent
        self.config = config
        self.responses = []
        # Create a Gradio chat interface connected to our agent interaction method
        self.app = gr.ChatInterface(
            self.interact_with_agent, 
            type="messages"
        )

    def interact_with_agent(self, user: str, history: List[Tuple[str, str]]) -> List[Dict[str, str]]:
        """Process user input and get responses from the agent.
        
        Args:
            user: The user's message text
            history: Chat history as maintained by Gradio
            
        Returns:
            A list of new messages from the agent
        """
        # Get messages from the agent's state before processing the new input
        msg = get_messages_from_langgraph_state(self.agent.get_state(self.config))
        before = list(convert_messages_from_langchain_to_gradio(msg))
        
        # Invoke the agent with the user's message
        self.responses = self.agent.invoke(
            {"messages": [{"role": "user", "content": user}]},
            config=self.config
        )
        
        # Get messages from the agent's state after processing
        msg = get_messages_from_langgraph_state(self.agent.get_state(self.config))
        after = list(convert_messages_from_langchain_to_gradio(msg))
        
        # Extract only the new messages that were added during this interaction
        # Skip the user's message (hence the +1) and return only agent responses
        new = after[len(before)+1:]
        return new

    def launch(self, **kwargs) -> None:
        """Launch the Gradio web interface.
        
        Args:
            **kwargs: Additional arguments to pass to gr.ChatInterface.launch()
        """
        self.app.launch(**kwargs)