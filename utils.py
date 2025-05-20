import requests
import os
import tempfile
import requests
import json
import re
from pathlib import Path
from typing import Optional, Tuple

from langchain_openai import ChatOpenAI
from langchain_deepseek import ChatDeepSeek
from openai import OpenAI

current_dir = Path(__file__).parent.absolute()
env_path = current_dir / ".env"

# read .config file
with open('.config', 'r') as f:
    config = json.load(f)

BASE_URL = config['BASE_URL']
DEBUG_MODE = config['DEBUG_MODE']

def check_api_keys():
    """Check for the presence of required API keys."""
    required_keys = ['OPENAI_API_KEY', 'DEEPSEEK_API_KEY', 'TAVILY_API_KEY', 'ANTHROPIC_API_KEY', 'GEMINI_API_KEY']
    missing_keys = [key for key in required_keys if not os.environ.get(key)]

    if missing_keys:
        return False
    else:
        return True

def setup_llm():
    """
    Setup the LLMs for the agent.
    """
    llm_agent_management = ChatDeepSeek(model="deepseek-chat", temperature=0)
    llm_question_decomposition = ChatDeepSeek(model="deepseek-chat", temperature=0)  # "deepseek-chat" / "deepseek-reasoner"
    # llm_question_analysis = ChatAnthropic(model="claude-3-7-sonnet-20250219", temperature=0)
    # llm_question_analysis = ChatGoogleGenerativeAI(model="gemini-2.0-flash", temperature=0)
    llm_tool_use = ChatDeepSeek(model="deepseek-chat", temperature=0)
    llm_vision = ChatOpenAI(model="gpt-4o", temperature=0)  # gemini-2.0-flash
    # llm_vision = ChatGoogleGenerativeAI(model="gemini-2.0-flash", temperature=0)
    openai_client = OpenAI()
    return llm_agent_management, llm_question_decomposition, llm_tool_use, llm_vision, openai_client
"""
def determine_file_type(file_data: bytes) -> str:
    try:
        magika = Magika()
        result = magika.identify_bytes(file_data)
        # Ensure the extension starts with a dot
        label = result.output.label
        if label:
            return f".{label}" if not label.startswith('.') else label
        else:
            return ".bin"  # Default binary extension
    except Exception as e:
        print(f"File type identification failed: {str(e)}")
        return ".unknown"
"""
def download_and_save_task_file(task_id: str, original_filename: str) -> Optional[str]:
    """
    Downloads a file associated with a task_id, uses the extension from
    original_filename, and saves it to a temporary directory.
    The saved filename will be task_id + extension_from_original_filename.

    Args:
        task_id: The ID of the task to download the file for.
        original_filename: The original filename from the task metadata.
                           The extension from this name will be used.

    Returns:
        The full path to the saved temporary file, or None if any step fails.
        The path to the file can be used as an input for the tools.
    """
    try:
        # 1. Download the file data
        url = f"{BASE_URL}/files/{task_id}"
        file_response = requests.get(url, timeout=20)
        file_response.raise_for_status()
        file_data = file_response.content
        if not file_data:
            print(f"No file data downloaded for task {task_id}")
            return None
        print(f"Downloaded associated file for task {task_id}")

        # 2. Determine the file extension solely from original_filename
        chosen_extension = ""
        if original_filename and isinstance(original_filename, str):
            name, ext = os.path.splitext(original_filename)
            if ext and ext != ".":  # Check if extension from original filename is valid
                chosen_extension = ext
            else:
                print(f"Warning: No valid extension found in original_filename ('{original_filename}') for task {task_id}. File will be saved without an extension in its name if task_id part also lacks one.")
        else:
            print(f"Warning: original_filename was not a valid string for task {task_id}. File may be saved without a proper extension.")
            
        # Ensure chosen_extension starts with a dot if it's not empty and doesn't already
        if chosen_extension and not chosen_extension.startswith('.'):
            chosen_extension = '.' + chosen_extension
        # If chosen_extension is still empty here, the file will be saved as 'task_id' (no explicit extension part added)

        # 3. Construct temporary file path
        temp_dir = tempfile.gettempdir()
        # The filename is task_id + the derived extension.
        temp_file_name = f"{task_id}{chosen_extension}"
        temp_file_path = os.path.join(temp_dir, temp_file_name)

        # 4. Save the file
        with open(temp_file_path, 'wb') as f:
            f.write(file_data)
        print(f"Saved remote file for task {task_id} to {temp_file_path}")
        return temp_file_path

    except requests.RequestException as e:
        print(f"Error downloading file for task {task_id}: {str(e)}")
        return None
    except Exception as e: # Catch other potential errors like issues with os.path.splitext if original_filename is weird
        print(f"Error processing or saving file for task {task_id}: {str(e)}")
        return None

def cleanup_temp_files(temp_file_path) -> None:
    """ Clean up temporary files created during processing. """
    try:
        # To be safer, ensure temp_file_path is indeed a Path object if Path.unlink() is to be used.
        # Or, if it's a string, os.remove(temp_file_path) is fine.
        # Assuming os.path.exists and os.remove for string paths as per original.
        if isinstance(temp_file_path, str) and temp_file_path.startswith(tempfile.gettempdir()) and os.path.exists(temp_file_path):
            os.remove(temp_file_path)
            print(f"Cleaned up temporary file: {temp_file_path}")
        elif isinstance(temp_file_path, Path) and str(temp_file_path).startswith(tempfile.gettempdir()) and temp_file_path.exists():
            temp_file_path.unlink()
            print(f"Cleaned up temporary file: {temp_file_path}")
    except Exception as e:
        print(f"Error cleaning up temp file {temp_file_path}: {str(e)}")

def process_file_for_task_v2(task_id: str, question_text: str, api_url: str) -> Tuple[str, Optional[Path]]:
    """
    Attempts to download a file for a task and appends its path to the question.
    Returns: (potentially modified question_text, path_to_downloaded_file or None)
    """
    file_download_url = f"{api_url}/files/{task_id}" 
    print(f"Attempting to download file for task {task_id} from {file_download_url}")
    local_file_path = None 

    try:
        response = requests.get(file_download_url, timeout=30)
        if response.status_code == 404:
            print(f"No file found for task {task_id} (404). Proceeding without file.")
            return question_text, None
        response.raise_for_status() # Raise an exception for other bad status codes (4xx, 5xx)
    except requests.exceptions.RequestException as exc:
        print(f"Error downloading file for task {task_id}: {exc}. Proceeding without file.")
        return question_text, None

    # Determine filename from 'Content-Disposition' header
    content_disposition = response.headers.get("content-disposition", "")
    # Adjusted regex to be more robust for quoted and unquoted filenames
    filename_match = re.search(r'filename="?([^"]+)"?', content_disposition) 
    
    filename_from_header = ""
    if filename_match:
        filename_from_header = filename_match.group(1)
    
    # Sanitize and ensure filename is not empty
    if filename_from_header:
        # A more robust sanitization might be needed depending on expected filenames
        # For now, replace non-alphanumeric (excluding ., _, -) with _
        filename = "".join(c if c.isalnum() or c in ('.', '_', '-') else '_' for c in filename_from_header).strip()
        if not filename: # If sanitization results in empty string or just spaces
            print(f"Warning: Sanitized filename from header for task {task_id} is empty. Using task_id as filename base.")
            filename = task_id
    else:
        print(f"Could not determine filename from Content-Disposition for task {task_id}. Using task_id as filename base.")
        filename = task_id

    # Ensure a reasonable default extension if none is apparent
    if '.' not in Path(filename).suffix: # Check if there's an extension part
        content_type = response.headers.get('Content-Type', '').split(';')[0].strip() # Get MIME type part
        extension = ""
        if content_type == 'image/jpeg': extension = '.jpg'
        elif content_type == 'image/png': extension = '.png'
        elif content_type == 'application/pdf': extension = '.pdf'
        elif content_type == 'text/plain': extension = '.txt'
        elif content_type == 'application/json': extension = '.json'
        elif content_type == 'text/csv': extension = '.csv'
        # Add more mime-type to extension mappings as needed
        
        if extension:
            filename += extension
        else:
            print(f"Warning: Could not determine extension for task {task_id} from Content-Type '{content_type}'. Using '.dat'.")
            filename += '.dat' # Generic data extension if type is unknown or unmapped

    temp_storage_dir = Path(tempfile.gettempdir()) / "hf_space_agent_files"
    temp_storage_dir.mkdir(parents=True, exist_ok=True)
    local_file_path = temp_storage_dir / Path(filename).name # Use Path(filename).name to ensure it's just the filename part

    try:
        with open(local_file_path, 'wb') as f:
            f.write(response.content)
        print(f"File for task {task_id} saved to: {local_file_path}")
        amended_question = (
            f"{question_text}\n\n"
            f"--- Technical Information ---\n"
            f"A file relevant to this task was downloaded and is available to your tools at the following local path. "
            f"Your tools that can read local files (like read_file, extract_text_from_image, etc.) should use this path:\n"
            f"Local file path: {str(local_file_path)}\n"
            f"--- End Technical Information ---\n\n"
        )
        return amended_question, local_file_path
    except IOError as e:
        print(f"Error saving file {local_file_path} for task {task_id}: {e}")
        return question_text, None # Saving failed



