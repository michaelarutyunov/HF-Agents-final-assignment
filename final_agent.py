# Standard libraries
import json
import os
from dotenv import load_dotenv
from typing import Dict, List, Any, Optional, Annotated
from typing_extensions import TypedDict

# Langchain and langgraph
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage, ToolMessage, AnyMessage
from langgraph.graph import StateGraph, START, END, add_messages
from langgraph.prebuilt import ToolNode, tools_condition

# Custom modules
from prompts import MAIN_SYSTEM_PROMPT, QUESTION_DECOMPOSITION_PROMPT, TOOL_USE_INSTRUCTION, EXECUTION_INSTRUCTION
from utils import check_api_keys, setup_llm
from tools import (
    calculator_tool, extract_text_from_image, transcribe_audio, execute_python_code,
    read_file, web_search, wikipedia_search, arxiv_search, chess_board_image_analysis,
    find_phrase_in_text, download_youtube_audio, web_content_extract, analyse_tabular_data
)

# AGENT STATE
class AgentState(TypedDict):
    task_id: Optional[str]
    file_name: Optional[str]
    file_type: Optional[str]
    file_path: Optional[str]
    question_decomposition: Optional[str]
    messages: Annotated[list[AnyMessage], add_messages]
    tool_results: Dict
    error_message: Optional[str]

# WORKFLOW CREATION
def create_workflow_for_final_agent():
    """
    Creates and compiles the LangGraph workflow.
    """
    llm_agent_management, llm_question_decomposition, _, _, _ = setup_llm()

    tools = [
        web_search,
        web_content_extract,
        wikipedia_search,
        calculator_tool,
        extract_text_from_image,
        transcribe_audio,
        execute_python_code,
        read_file,
        arxiv_search,
        chess_board_image_analysis,
        find_phrase_in_text,
        download_youtube_audio,
        analyse_tabular_data
    ]

    llm_agent_management_with_tools = llm_agent_management.bind_tools(tools)

    # Define nodes
    def question_decomposition_node(state: AgentState):
        new_state = state.copy()
        messages = new_state.get("messages", []) # Use .get for safety, ensure it's a list
        question = None
        for msg in messages:
            if isinstance(msg, HumanMessage):
                question = msg.content
                break
        if not question:
            new_state["error_message"] = "No question found for decomposition."
            # Ensure messages list exists even if we return early
            if "messages" not in new_state or not isinstance(new_state["messages"], list):
                new_state["messages"] = [] 
            return new_state

        question_decomposition_prompt_messages = [
            SystemMessage(content=QUESTION_DECOMPOSITION_PROMPT),
            HumanMessage(content=f"Decompose this question: {question}")
        ]
        question_decomposition_object = llm_question_decomposition.invoke(question_decomposition_prompt_messages)
        question_decomposition_response = question_decomposition_object.content
        new_state["question_decomposition"] = question_decomposition_response
        # Ensure messages list exists
        if "messages" not in new_state or not isinstance(new_state["messages"], list):
             new_state["messages"] = []
        return new_state

    def call_model_node(state: AgentState):
        new_state = state.copy()
        messages = new_state.get("messages", []) # Use .get for safety
        question_decomposition = new_state.get("question_decomposition", "")

        llm_messages = list(messages) # Ensure it's a mutable list
        
        add_decomposition = question_decomposition and (not llm_messages or not isinstance(llm_messages[-1], ToolMessage))
        if add_decomposition:
            decomposition_message = SystemMessage(content=f"Question decomposition: {question_decomposition}\\nUse this analysis to guide your actions.")
            llm_messages.append(decomposition_message)

        response = llm_agent_management_with_tools.invoke(llm_messages)
        
        # Ensure new_state["messages"] exists and is a list before extending
        current_messages = new_state.get("messages", [])
        if not isinstance(current_messages, list):
            current_messages = []
        new_state["messages"] = current_messages + [response]
        return new_state

    workflow = StateGraph(AgentState)
    workflow.add_node("decomposition", question_decomposition_node)
    workflow.add_node("agent", call_model_node)
    workflow.add_node("tools", ToolNode(tools))

    workflow.add_edge(START, "decomposition")
    workflow.add_edge("decomposition", "agent")
    workflow.add_conditional_edges("agent", tools_condition)
    workflow.add_edge("tools", "agent")

    return workflow.compile()


class FinalAgent:
    def __init__(self):
        print("FinalAgent initializing...")
        load_dotenv()

        if not os.path.exists('.config'):
            print("Warning: .config file not found. Using default values or expecting environment variables.")
            self.config = {} # Default to empty config
        else:
            with open('.config', 'r') as f:
                self.config = json.load(f)
        
        self.base_url = self.config.get('BASE_URL', os.getenv('BASE_URL'))
        self.debug_mode = self.config.get('DEBUG_MODE', str(os.getenv('DEBUG_MODE', 'False')).lower() == 'true')

        if not check_api_keys():
            # check_api_keys itself prints messages
            raise ValueError("API keys are missing or invalid. Please set the required environment variables.")

        self.workflow = create_workflow_for_final_agent()
        print("FinalAgent initialized successfully.")

    def __call__(self, question: str, task_id: Optional[str] = None) -> str:
        print(f"FinalAgent received question for task_id '{task_id}': {question[:100]}...")

        initial_messages = [
            SystemMessage(content=MAIN_SYSTEM_PROMPT + "\\n\\n" + TOOL_USE_INSTRUCTION + "\\n\\n" + EXECUTION_INSTRUCTION),
            HumanMessage(content=question)
        ]
        
        initial_state: AgentState = {
            "messages": initial_messages,
            "task_id": task_id,
            "file_name": None,
            "file_path": None,
            "file_type": None,
            "question_decomposition": None, 
            "tool_results": {},
            "error_message": None
        }

        try:
            result_state = self.workflow.invoke(initial_state)
        except Exception as e:
            print(f"Error invoking workflow for task {task_id}: {e}")
            import traceback
            traceback.print_exc()
            return f"AGENT ERROR: Failed to process question due to an internal error: {e}"

        messages = result_state.get("messages", [])
        final_answer = ""
        if not messages:
            print(f"No messages found in the result state for task {task_id}.")
            return "AGENT ERROR: No messages returned by the agent."

        for msg in reversed(messages):
            if hasattr(msg, "content") and msg.content:
                content = msg.content
                if isinstance(content, str):
                    if "FINAL ANSWER:" in content:
                        final_answer = content.split("FINAL ANSWER:", 1)[1].strip()
                        break 
                    elif isinstance(msg, AIMessage):
                        # If it's an AIMessage and no "FINAL ANSWER:" has been found yet,
                        # tentatively set it. This will be overridden if a "FINAL ANSWER:" is found later.
                        if not final_answer: 
                            final_answer = content
        
        # If after checking all messages, final_answer is still from a non-"FINAL ANSWER:" AIMessage, that's our best guess.
        # If final_answer is empty, it means no AIMessage with content or "FINAL ANSWER:" was found.
        if not final_answer: # This means no "FINAL ANSWER:" and no AIMessage content was suitable
            final_answer = "AGENT ERROR: Could not extract a final answer from the agent's messages."
            print(f"Could not extract final answer for task {task_id}. Messages: {messages}")

        print(f"FinalAgent returning answer for task_id '{task_id}': {final_answer[:100]}...")
        return final_answer

if __name__ == '__main__':
    print("Running a simple test for FinalAgent...")
    
    if not os.path.exists('.config'):
        print("Creating a dummy .config file for testing.")
        with open('.config', 'w') as f:
            json.dump({"DEBUG_MODE": True, "BASE_URL": "http://localhost:8000"}, f)

    # Check for .env and API keys
    if not load_dotenv(): # Attempts to load .env and returns True if successful
        print("Warning: .env file not found or failed to load. API keys might be missing.")

    if not (os.getenv("OPENAI_API_KEY") or os.getenv("ANTHROPIC_API_KEY") or os.getenv("GEMINI_API_KEY")):
        print("\\nWARNING: No major LLM API key found in environment variables (OPENAI_API_KEY, ANTHROPIC_API_KEY, GEMINI_API_KEY).")
        print("The agent will likely fail to initialize or run properly without at least one.")
        print("Please set them in your .env file or environment for testing.\\n")

    try:
        agent = FinalAgent()
        test_question = "What is the capital of France? And what is the weather like there today?"
        print(f"Test Question 1: {test_question}")
        answer = agent(test_question, task_id="test_001")
        print(f"Test Answer 1: {answer}")

        test_question_calc = "What is 123 * 4 / 2 + 6?"
        print(f"\\nTest Question 2 (Calc): {test_question_calc}")
        answer_calc = agent(test_question_calc, task_id="test_002")
        print(f"Test Answer 2 (Calc): {answer_calc}")

    except ValueError as ve:
        print(f"Initialization Error: {ve}")
    except Exception as e:
        print(f"An error occurred during the test: {e}")
        import traceback
        traceback.print_exc() 