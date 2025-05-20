# Standard libraries
import argparse
import json
import logging
import os
from dotenv import load_dotenv
from typing_extensions import TypedDict
from typing import Dict, List, Any, Optional, Annotated

# Langchain and langgraph
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage,  ToolMessage, AnyMessage
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition

# Custom modules
from prompts import MAIN_SYSTEM_PROMPT, QUESTION_DECOMPOSITION_PROMPT, TOOL_USE_INSTRUCTION, EXECUTION_INSTRUCTION
from utils import check_api_keys, setup_llm, download_and_save_task_file, cleanup_temp_files
from tools import calculator_tool, extract_text_from_image, transcribe_audio, execute_python_code, read_file, web_search, wikipedia_search, arxiv_search, chess_board_image_analysis, find_phrase_in_text, download_youtube_audio, web_content_extract, analyse_tabular_data, csv_reader

# SETUP
logging.basicConfig(
    level=logging.INFO,  # Set default level to INFO
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    filename='agent_run.log',
    filemode='w',  # 'a' to append to the file on each run, 'w' to overwrite
    )

load_dotenv()

with open('.config', 'r') as f:
    config = json.load(f)
BASE_URL = config['BASE_URL']
DEBUG_MODE = config['DEBUG_MODE']

# AGENT STATE
class AgentState(TypedDict):
    task_id: str
    file_name: Optional[str]
    file_type: Optional[str]
    file_path: Optional[str]
    question_decomposition: Optional[str]
    messages: Annotated[list[AnyMessage], add_messages]
    # tool_calls: Optional[List[str]] # Note: This field doesn't seem to be explicitly used or populated in the current code.
    tool_results: Dict  # Results from tool executions (populated implicitly by ToolNode?)
    error_message: Optional[str]

# WORKFLOW SETUP
def create_workflow(): 

    # Setup LLMs and tools
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
        analyse_tabular_data,
        csv_reader
        ]
    
    llm_agent_management_with_tools = llm_agent_management.bind_tools(tools)

    # Define nodes    
    def question_decomposition(state: AgentState):
        """Analyze question and decompose it into a plan to answer it. The plan will be used to guide the agent's actions."""
        new_state = state.copy() # Create a copy of the current state
        messages = new_state["messages"] # Get the messages from the current state

        # Find the HumanMessage to analyze
        question = None
        for msg in messages:
            if isinstance(msg, HumanMessage):
                question = msg.content
                break

        if not question:
            new_state["error_message"] = "No question found in the state for question_decomposition node"
            return new_state  # Return copy of original state unchanged

        # Create and invoke the analysis prompt
        question_decomposition_prompt = [
            SystemMessage(content=QUESTION_DECOMPOSITION_PROMPT),
            HumanMessage(content=f"Decompose this question : {question}")
        ]

        question_decomposition_object = llm_question_decomposition.invoke(question_decomposition_prompt)
        question_decomposition_response = question_decomposition_object.content

        # Update the state with the question analysis
        new_state["question_decomposition"] = question_decomposition_response

        # Return the complete new state with all fields preserved
        return new_state

    def call_model(state: AgentState):
        """Invoke the LLM with the current state."""
        new_state = state.copy()
        messages = new_state["messages"]
        question_decomposition = new_state.get("question_decomposition", "")
        
        # Prepare messages for the LLM call
        llm_messages = messages.copy()
        add_decomposition = question_decomposition and (not messages or not isinstance(messages[-1], ToolMessage))

        if add_decomposition:
            decomposition_message = SystemMessage(content=f"Question decomposition: {question_decomposition}\nUse this analysis to guide your actions.")
            llm_messages.append(decomposition_message)

        response = llm_agent_management_with_tools.invoke(llm_messages)
        new_state["messages"] = messages + [response] 
        return new_state
    """
    def router(state: AgentState):
        #Determine whether to continue to tools or end the workflow.
        messages = state["messages"]
        last_message = messages[-1]

        if last_message.tool_calls:
            return "tools"
        return END
    """

    # Setup workflow
    workflow = StateGraph(AgentState)

    workflow.add_node("decomposition", question_decomposition)
    workflow.add_node("agent", call_model)
    workflow.add_node("tools", ToolNode(tools))

    workflow.add_edge(START, "decomposition")
    workflow.add_edge("decomposition", "agent")
    workflow.add_conditional_edges("agent", tools_condition)  # router, {"tools": "tools", END: END}
    workflow.add_edge("tools", "agent")

    return workflow.compile()

# REPORTING
# save the messages to a txt file
def save_txt_report(state: AgentState, task_id: str):
    """Create a txt report from the state."""
    messages = state["messages"]
    report = ""

    # question wording
    question = messages[0].content
    report += f"Question: {question}\n\n"

    # question decomposition
    question_decomposition = state.get("question_decomposition", "No decomposition available")
    report += f"Question decomposition: {question_decomposition}\n\n"

    # message content
    report += "Message Chain:\n"
    for msg in messages:
        msg_type = type(msg).__name__ # Get the class name (e.g., "HumanMessage")
        report += f"--- {msg_type} ---\n"
        report += f"{msg.content}\n"
        # Optionally add tool call info for AIMessages
        if hasattr(msg, 'tool_calls') and msg.tool_calls:
             report += f"Tool Calls: {msg.tool_calls}\n"
        # ToolMessage content often includes tool output directly,
        # but you could format it differently if needed.
        report += "---\n\n"

    # find the task with matching task_id
    validation_data = []
    correct_answer = ""

    with open("metadata_val.jsonl", "r", encoding="utf-8") as file:
        for line in file:
            validation_data.append(json.loads(line))

    for task_metadata in validation_data:
        if task_metadata.get("task_id") == task_id:
            correct_answer = task_metadata.get("Final answer", "Not found")

    report += f"Correct answer: {correct_answer}"

    with open(f"text_report_{task_id}.txt", "w", encoding="utf-8") as f:
        f.write(report)

    return report

# WORKFLOW EXECUTION
def execute_workflow(tasks_file: str, output_file: str):
    
    with open(tasks_file, 'r', encoding='utf-8') as f:
        tasks = json.load(f)

    results_json = [] 

    # iterate over tasks
    for task in tasks:
        task_id = task["task_id"]
        question = task["question"]
        temp_file_to_cleanup = None

        # prepare content for HumanMessage
        if task.get("file_name"):
            original_filename_from_task = task["file_name"]
            temp_file_path = download_and_save_task_file(task_id, original_filename_from_task)

            if temp_file_path:
                temp_file_to_cleanup = temp_file_path

                # Construct the question, relying on the LLM to infer file type from the path's extension
                question = task["question"] + f"\n\nAttached file: {temp_file_path}"
                print(f"File for task {task_id} processed and available at: {temp_file_path}")
            else:
                print(f"Failed to download or save file for task {task_id} using filename '{original_filename_from_task}'. Proceeding with question only.")

        # run agent
        try:
            print(f"Running agent for task {task_id}")
            print(f"Question: {question}")

            workflow = create_workflow()

            result = workflow.invoke({
                "messages": [
                    SystemMessage(content=MAIN_SYSTEM_PROMPT + "\n\n" + TOOL_USE_INSTRUCTION + "\n\n" + EXECUTION_INSTRUCTION),
                    HumanMessage(content=question)
                ]
            })

            # Extract final answer - result is the state itself with messages
            messages = result.get("messages", [])
            final_answer = ""
            
            # Get the content from the last message that has content
            for msg in reversed(messages):
                if hasattr(msg, "content") and msg.content:
                    content = msg.content
                    # Extract answer using the template format
                    if "FINAL ANSWER:" in content:
                        final_answer = content.split("FINAL ANSWER:")[1].strip()
                    else:
                        final_answer = content
                    break
            
            if not final_answer:
                final_answer = "No answer generated"

            # Extract question decomposition
            question_decomposition = result.get("question_decomposition", "")

            # Save results to results.json
            if DEBUG_MODE:
                validation_data = []
                with open("metadata_val.jsonl", "r", encoding="utf-8") as file:
                    for line in file:
                        validation_data.append(json.loads(line))

                correct_answer = "Not found"
                for task_metadata in validation_data:
                    if task_metadata.get("task_id") == task_id:
                        correct_answer = task_metadata.get("Final answer", "Not found")
                        break
                results_json.append({"task_id": task_id, "model_answer": final_answer, "correct_answer": correct_answer, "question_decomposition": question_decomposition})
            else:
                results_json.append({"task_id": task_id, "model_answer": final_answer, "question_decomposition": question_decomposition})

            # Save state to txt report
            if DEBUG_MODE:
                save_txt_report(result, task_id)

        except Exception as e:
            print(f"Error processing task {task_id}: {str(e)}")
            import traceback
            traceback.print_exc()

        finally:
            if temp_file_to_cleanup and os.path.exists(temp_file_to_cleanup):
                cleanup_temp_files(temp_file_to_cleanup) 
            elif task.get("file_name") and not temp_file_to_cleanup:
                 print(f"Note for task {task_id}: A file was expected, but no temporary file was successfully processed or tracked for cleanup.")

    # save batch results
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results_json, f, indent=2)

    return print(f"Results saved to {output_file}")

# MAIN
def main():
    parser = argparse.ArgumentParser(description='Process tasks from a JSON file and save results to an output file.')
    parser.add_argument('--tasks-file', type=str, default='tasks.json', help='Path to the JSON file containing tasks')
    parser.add_argument('--output-file', type=str, default='results.json', help='Path to the output JSON file')
    args = parser.parse_args()

    if not check_api_keys():
        print("API keys are missing. Please set the required environment variables.")
        return

    try:
        execute_workflow(args.tasks_file, args.output_file)
    except Exception as e:
        print(f"Critical error: {str(e)}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    main()