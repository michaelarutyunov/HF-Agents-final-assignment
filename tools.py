import base64
import csv
from langchain_openai import ChatOpenAI
import openai
import requests
import pandas as pd
import tempfile
import os
import json
from pathlib import Path
from typing import Any, Union
from dotenv import load_dotenv
from bs4 import BeautifulSoup


from langchain_core.messages import HumanMessage
from langchain_core.output_parsers import StrOutputParser
from langchain.tools import tool
from langchain_openai import ChatOpenAI
from langchain_community.document_loaders import YoutubeLoader
from langchain_community.document_loaders.youtube import TranscriptFormat
from langchain_community.tools import ArxivQueryRun
from langchain_community.utilities import ArxivAPIWrapper
from langchain.agents.agent_types import AgentType
from langchain_experimental.agents.agent_toolkits import create_pandas_dataframe_agent

import wikipedia
from langchain_tavily import TavilySearch
import yt_dlp
from yt_dlp.utils import DownloadError, ExtractorError
from utils import setup_llm

current_dir = Path(__file__).parent.absolute()
env_path = current_dir / ".env"

load_dotenv(dotenv_path=env_path, override=True)

@tool
def read_file(file_path: str) -> str:
    """Read the entire content of a text file specified by its path."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return f.read()
    except Exception as e:
        return f"read_file tool error: {str(e)}"

@tool
def web_search(query: str) -> str:
    """
    Searches the web and returns a list of the most relevant URLs.
    Use this FIRST for complex queries, metadata questions, or to find the right sources.
    Then follow up with web_content_extract on the most promising URL.
    """
    try:
        tavily_search = TavilySearch(
            max_results=5,
            topic="general",
            search_depth="advanced",
            include_raw_content=False,  # Just URLs and snippets
        )

        results = tavily_search.invoke(query)
        # Format results to show URLs and brief descriptions
        web_search_results = "Search Results:\n"
        for i, result in enumerate(results["results"], 1):
            web_search_results += f"{i}. {result['title']}: {result['url']}\n   {result['content'][:150]}...\n\n"

        return web_search_results
    except Exception as e:
        return f"web_search tool error: {str(e)}"

@tool
def web_content_extract(url: str) -> str:
    """
    Extracts and analyzes specific content from a URL using BeautifulSoup.
    Particularly effective for Wikipedia metadata pages, discussion pages, 
    and structured web content.
    Can be used after web_search to get detailed information.
    """
    try:
        import requests
        from bs4 import BeautifulSoup

        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
        }

        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()  # Raise exception for 4XX/5XX responses

        soup = BeautifulSoup(response.text, 'html.parser')
        for element in soup.select('script, style, footer, nav, header'):
            if element:
                element.decompose()          
        text = soup.body.get_text(separator='\n', strip=True) if soup.body else soup.get_text(separator='\n', strip=True)

        # Limit content length for response
        return f"Content extracted from {url}:\n\n{text[:10000]}..." if len(text) > 10000 else text

    except Exception as e:
        return f"web_content_extract tool error: {str(e)}"
"""
# not used
def web_search(query: str) -> str:
    
    Searches the web for general information. Optimal for:
    1. Recent events or time-sensitive information
    2. Metadata about websites
    3. Questions combining multiple specific criteria
    4. Queries that need to search across multiple websites or data sources
    
    try:
        tavily_search = TavilySearch(
            max_results=3,
            topic="general",
            search_depth="basic",
            include_raw_content=True
        )

        # This returns a string representation of the results
        return tavily_search.invoke(query)
    except Exception as e:
        return f"web_search tool error: {str(e)}"
    _summary_

    Raises:
        ValueError: _description_
        RuntimeError: _description_

    Returns:
        _type_: _description_
"""
@tool
def wikipedia_search(query: str, get_summary: bool = True) -> str:
    """
    Searches Wikipedia for factual information contained within articles.
    Best for:
    1. Basic encyclopedic facts about topics
    2. Definitions and explanations of concepts
    3. Historical information about well-documented subjects
    4. Classification and categorical information

    Args:
        query (str): The query to search Wikipedia for
        get_summary (bool): Whether to get the summary of the Wikipedia page instead of the full content
    """

    try:
        page = wikipedia.page(title=query, auto_suggest=True, redirect=True)

        # text_content = page.content  # excluding images, tables, and other data
        full_content_html = page.html()

        # parse the html content
        soup = BeautifulSoup(full_content_html, 'html.parser')
        text_content = soup.get_text()

        if get_summary:
            llm_agent_management = setup_llm()[0]
            message = [
                HumanMessage(content=f"Provide response to the following query: {query}\n\nbased on the following content: {text_content}")
            ]
            response = llm_agent_management.invoke(message)
            return response.content
        
        # summary = page.summary
        response = f"Page: {page.title}\nSource: {page.url}\n\n{text_content}"
        if response:
            return StrOutputParser().invoke(response)
        else:
            return "wikipedia_search tool produced empty response"

    # Basic error handling for wikipedia library issues
    except wikipedia.exceptions.PageError:
        # Use the original query in the error message as 'page' might not be defined
        return f"Wikipedia page for query '{query}' does not match any known pages."
    except wikipedia.exceptions.DisambiguationError as e:
        # Provide options if it's a disambiguation page
        options = "\n - ".join(e.options[:5])  # Show first 5 options
        return f"Ambiguous query '{query}'. Did you mean:\n - {options}\nPlease refine your search."
    except Exception as e:
        return f"wikipedia_search tool error: {str(e)}"

@tool
def arxiv_search(query: str) -> str:
    """
    Search Arxiv for scientific papers
    """
    try:
        arxiv_api = ArxivAPIWrapper(top_k_results=3)
        arxiv_search = ArxivQueryRun(api_wrapper=arxiv_api)
        result = arxiv_search.invoke(query)
        response = result.content + "\n\n"

        if response:
            return response
        else:
            return "arxiv_search tool produced empty response"
    except Exception as e:
        return f"arxiv_search tool error: {str(e)}"

@tool
def calculator_tool(expression: str) -> str:
    """Evaluate a mathematical expression."""
    try:
        result = eval(expression, {"__builtins__": {}},
                      {"abs": abs, "round": round, "max": max, "min": min})
        if result:
            return str(result)
        else:
            return "calculator tool produced empty response"
    except Exception as e:
        return f"calculator tool error: {str(e)}"

@tool
def extract_text_from_image(image_path: str) -> str:
    """Extract text from a locally saved image."""

    try:
        llm_vision = setup_llm()[3]
        with open(image_path, "rb") as image_file:
            image_bytes = image_file.read()
            image_base64 = base64.b64encode(image_bytes).decode("utf-8")

            message = [
                HumanMessage(
                    content=[
                        {
                            "type": "text",
                            "text": (
                                "Extract all the text from this image. "
                                "Return only the extracted text, no explanations."
                            ),
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/png;base64,{image_base64}"
                            },
                        },
                    ]
                )
            ]

        result = llm_vision.invoke(message)
        response = result.content + "\n\n"

        if response:
            return response
        else:
            return "extract_text_from_image tool produced empty response"
    except Exception as e:
        return f"extract_text_from_image tool error: {str(e)}"

@tool
def extract_youtube_video(video_url: str) -> str:
    """Extract text from a youtube video."""
    try:
        loader = YoutubeLoader.from_youtube_url(
            video_url,
            transcript_format=TranscriptFormat.TEXT,
            language="en",
            add_video_info=True,
        )

        result = loader.load()
        response = result[0].page_content + "\n\n"

        if response:
            return response
        else:
            return "extract_youtube_video tool produced empty response"
    except Exception as e:
        return f"extract_youtube_video tool error: {str(e)}"

@tool
def transcribe_audio(audio_path: str) -> str:
    """Extract text from a locally saved audio file."""
    
    try:
        openai_client = setup_llm()[4]

        with open(audio_path, "rb") as audio_file:
            response = openai_client.audio.transcriptions.create(
                model="gpt-4o-mini-transcribe",
                file=audio_file
            )

            if response:
                return response.text
            else:
                return "transcribe_audio tool produced empty response"
    except Exception as e:
        return f"transcribe_audio tool error: {str(e)}"

@tool
def query_about_image(query: str, image_url: str) -> str:
    """Ask anything about an image from a URL using a Vision Language Model
    Args:
        query (str): The query about the image
        image_url (str): The URL to the image
    """

    try:
        openai_client = setup_llm()[4]
        response = openai_client.responses.create(
            model="gpt-4o-mini",
            input=[{
                "role": "user",
                "content": [
                    {"type": "input_text", "text": query},
                    {
                        "type": "input_image",
                        "image_url": image_url,
                    },
                ],
            }],
        )

        if response:
            return response.text
        else:
            return "query_about_image tool produced empty response"
    except Exception as e:
        return f"query_about_image tool error: {str(e)}"

@tool
def execute_python_code(code: str) -> Any:
    """Executes Python code safely in a restricted environment."""
    
    try:
        # Basic restricted execution
        allowed_modules = {'math', 're', 'json'}
        forbidden_terms = ['import', 'exec', 'eval', 'open', 'os', 'sys', 'subprocess']

        # Simple check for forbidden terms
        for term in forbidden_terms:
            if term in code:
                return f"Forbidden term used: {term}"

        # Create a restricted globals dictionary
        restricted_globals = {
            '__builtins__': {
                k: __builtins__[k] for k in __builtins__
                if k not in ['open', 'exec', 'eval', 'compile']
            }
        }

        # Allowed modules
        for module_name in allowed_modules:
            restricted_globals[module_name] = __import__(module_name)

        # Execute the code with restricted globals and locals
        local_vars = {}
        exec(code, restricted_globals, local_vars)

        # Return the local variables after execution
        return local_vars

    except Exception as e:
        return f"Code execution error: {str(e)}"

@tool
def csv_reader(file_path: str) -> str:
    """
    Read a csv file and return a string of the data.

    Args:
        file_path (str): The path to the csv file

    Returns:
        str: A string of the data in the csv file
    """
    try:
        df = pd.read_csv(file_path, encoding='utf-8')

        if df.empty:
            return f"CSV file found at '{file_path}' but it is empty."

        return df.to_string()

    except FileNotFoundError:
        return f"csv_reader tool error: File not found at '{file_path}'"
    except pd.errors.EmptyDataError:
        return f"csv_reader tool error: No data found in CSV file '{file_path}'."
    except pd.errors.ParserError as e:
        return f"csv_reader tool error: Error parsing CSV file '{file_path}'. Details: {str(e)}"
    except Exception as e:
        # Catch any other unexpected errors
        import traceback
        tb_str = traceback.format_exc()
        return f"csv_reader tool error: An unexpected error occurred while reading '{file_path}'. Error: {str(e)}\nTraceback:\n{tb_str}"

@tool
def chess_board_image_analysis(image_path, order_of_play: str = "black") -> str:
    """Analyze a chess position from an image and order of play (black or white) and return the best move in algebraic notation."""
    import chess
    import base64
    import requests
    import os.path

    def image_to_chess_json(image_path: str) -> Union[dict, str]:
        try:
            llm_vision = setup_llm()[3]
            with open(image_path, "rb") as image_file:
                image_bytes = image_file.read()
                image_base64 = base64.b64encode(image_bytes).decode("utf-8")

                message = [
                    HumanMessage(
                        content=[
                            {
                                "type": "text",
                                "text":
                                    """Analyze this image of a chessboard and return the position of each figure in the following format:
                                    {
                                        "figure_name": "position_on_board"
                                    }
                                    
                                    Important instructions:
                                    1. Each figure on the board is represented by a unique line in the JSON object
                                    2. Return ONLY the raw JSON without any formatting, markdown, code blocks, or explanations
                                    3. Do not use triple backticks
                                    4. Do not include the string "json" before the output
                                    5. Just return the plain JSON object directly

                                    Verification: number of records in the JSON object should be equal to the number of figures on the board.
                                    """,
                            },
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/png;base64,{image_base64}"
                                },
                            },
                        ]
                    )
                ]

            result = llm_vision.invoke(message)
            response = result.content

            try:
                position_dict = json.loads(response)
                print(position_dict)
                return position_dict
            except json.JSONDecodeError:
                return f"Error: Could not parse response as JSON: {response}"
        except Exception as e:
            return f"image_to_fen_llm tool error: {str(e)}"

    def create_fen_from_position(position_dict):
        """Convert a position dictionary to FEN notation."""
        # Initialize an 8x8 empty board
        board = [['' for _ in range(8)] for _ in range(8)]

        # Map of piece names to FEN characters
        piece_map = {
            'white_king': 'K', 'white_queen': 'Q',
            'white_rook': 'R', 'white_bishop': 'B',
            'white_knight': 'N', 'white_pawn': 'P',
            'black_king': 'k', 'black_queen': 'q',
            'black_rook': 'r', 'black_bishop': 'b',
            'black_knight': 'n', 'black_pawn': 'p'
        }

        # Place pieces on the board
        for piece, position in position_dict.items():
            if not position or len(position) < 2:
                continue  # Skip invalid positions

            # Convert UCI notation to board indices
            file, rank = position[0], position[1]
            col = ord(file) - ord('a')
            row = 8 - int(rank)

            # Skip if out of board bounds
            if col < 0 or col > 7 or row < 0 or row > 7:
                continue

            # Determine the correct piece symbol
            piece_symbol = ''

            # Direct mapping if piece name exactly matches
            if piece in piece_map:
                piece_symbol = piece_map[piece]
            else:
                # Strip numeric suffix if present (e.g., white_pawn1 -> white_pawn)
                base_name = piece.rstrip('0123456789')
                if base_name.endswith('_'):
                    base_name = base_name[:-1]

                if base_name in piece_map:
                    piece_symbol = piece_map[base_name]
                else:
                    # Try partial matching by looking for key prefixes
                    for key, symbol in piece_map.items():
                        if piece.startswith(key):
                            piece_symbol = symbol
                            break

            # Place the piece on the board
            if piece_symbol:
                board[row][col] = piece_symbol

        # Convert board to FEN piece placement notation
        fen_parts = []
        for row in board:
            empty_count = 0
            rank_str = ""

            for cell in row:
                if cell == '':
                    empty_count += 1
                else:
                    if empty_count > 0:
                        rank_str += str(empty_count)
                        empty_count = 0
                    rank_str += cell

            if empty_count > 0:
                rank_str += str(empty_count)

            fen_parts.append(rank_str)

        piece_placement = '/'.join(fen_parts)

        active_color = "b" if order_of_play == "black" else "w"
        castling = "KQkq" # Some chess specific parameters that i do not understand
        en_passant = "-"
        halfmove_clock = "0"
        fullmove_number = "1"

        # Construct the complete FEN string
        fen_notation = f"{piece_placement} {active_color} {castling} {en_passant} {halfmove_clock} {fullmove_number}"

        return fen_notation
    
    def get_best_move(fen_notation: str) -> str:
        """Get the best move from Stockfish chess engine API."""
        if not fen_notation:
            return "Error: Invalid FEN notation"

        try:
            api_url_stockfish = "https://stockfish.online/api/s/v2.php" # this is that specific chess API
            depth = 10 # depth of analysis, i believe normally between 10-15
            
            import urllib.parse
            encoded_fen = urllib.parse.quote(fen_notation)
            full_url = f"{api_url_stockfish}?fen={encoded_fen}&depth={depth}"

            response = requests.get(full_url, timeout=60)
            
            if response.status_code == 200:
                result = response.json()
                if result.get("success", False):
                    best_move = result.get("bestmove", "")

                    # The API returns format like "bestmove e2e4 ponder h7h5" and only move part is needed
                    if " " in best_move:
                        best_move = best_move.split(" ")[1]  # Get the actual move

                    return best_move
                else:
                    return f"Failed to get best move: {result.get('data', 'Unknown error')}"

            return f"Failed to get best move. Status code: {response.status_code}"
        except Exception as e:
            return f"Error getting best move: {str(e)}"

    def convert_uci_to_algebraic(fen_notation, uci_move):
        """Convert UCI move notation to algebraic notation."""
        if not fen_notation or not uci_move:
            return "Error: Missing FEN notation or UCI move"

        try:
            # Here comes input from step 2
            board = chess.Board(fen_notation)

            # Here comes input from step 3
            move = chess.Move.from_uci(uci_move)

            # Verify move is legal
            if move not in board.legal_moves:
                return f"Error: {uci_move} is not a legal move in this position"

            # Get algebraic notation
            algebraic_move = board.san(move)
            return algebraic_move
        except ValueError as e:
            return f"Error converting move: {str(e)}"
        except Exception as e:
            return f"Unexpected error: {str(e)}"

    # Main function logic
    try:
        # Get board JSON from image
        chess_json = image_to_chess_json(image_path)
        if isinstance(chess_json, str) and (chess_json.startswith("Error") or chess_json.startswith("Failed")):
            return chess_json

        # Get FEN notation from board JSON
        fen_notation = create_fen_from_position(chess_json)

        # Pass FEN notation to chess API and get best move in UCI format
        uci_move = get_best_move(fen_notation)
        if isinstance(uci_move, str) and (uci_move.startswith("Error") or uci_move.startswith("Failed")):
            return uci_move

        # Convert UCI format response to algebraic notation
        algebraic_move = convert_uci_to_algebraic(fen_notation, uci_move)

        # Return the result
        if algebraic_move.startswith("Error"):
            return algebraic_move
        else:
            return f"Best move: {algebraic_move}"
    except Exception as e:
        return f"Chess board analysis failed: {str(e)}"

@tool
def download_youtube_audio(url, task_id):
    """Download the audio from a YouTube video"""
    temp_dir = tempfile.gettempdir()
    output_filename_template = os.path.join(temp_dir, f"{task_id}.%(ext)s")
    downloaded_audio_file_path = os.path.join(temp_dir, f"{task_id}.mp3")
    
    ydl_opts = {
        'format': 'bestaudio/best',  # Select best audio quality available
        'outtmpl': output_filename_template,  # Temporary filename pattern
        'postprocessors': [{
            'key': 'FFmpegExtractAudio',  # Use FFmpeg to extract audio
            'preferredcodec': 'mp3',      # Convert to MP3 format
            'preferredquality': '192',    # Set audio quality (bitrate)
        }],
        'quiet': True,  # Suppress console output
    }

    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            ydl.download([url])
            if os.path.exists(downloaded_audio_file_path):
                print(f"Successfully downloaded and converted audio to: {downloaded_audio_file_path}")
                return downloaded_audio_file_path
            else:
                print(f"Failed to download audio from {url}")
                return None

    except DownloadError as e:
        print(f"yt-dlp download error for {url} (Task ID: {task_id}): {e}")
        return None
    except ExtractorError as e:
        print(f"yt-dlp extractor error for {url} (Task ID: {task_id}): {e}")
        return None
    except Exception as e:
        print(f"An unexpected error occurred while processing {url} (Task ID: {task_id}): {e}")
        return None

@tool
def find_phrase_in_text(text, phrase):
    """
    Find a specific phrase in the text and return its segment with the next segment
    
    Args:
        text (str): The text to search
        phrase (str): The phrase to look for
        
    Returns:
        tuple: (question segment, response segment)
    """
    segments = text['segments']

    # Convert the question phrase to lowercase for case-insensitive matching
    phrase_lower = phrase.lower()

    # Find the segment containing the question
    phrase_segment = None

    for i, segment in enumerate(segments):
        if phrase_lower in segment['text'].lower():
            phrase_segment = segment
            # If we found the question, the response is likely in the next segment
            if i + 1 < len(segments):
                response_segment = segments[i + 1]
                return phrase_segment, response_segment

    return None, None

@tool
def analyse_tabular_data(table_path, query: str) -> str:
    """
    Read a table and return the answer to a question.

    Args:
        table_path (str): The path to the table
        query (str): The question to answer

    Returns:
        str: The answer to the question
    """
    try:
         # Read the table
        file_type = table_path.split(".")[-1]
        reader_map = {
        "csv": pd.read_csv,
        "json": pd.read_json,
        "xlsx": pd.read_excel,
        "xls": pd.read_excel,
        }

        if file_type not in reader_map:
            return f"Unsupported file type: {file_type}"

        df = reader_map[file_type](table_path)
    except Exception as e:
        return f"Error reading table: {str(e)}"

    if df is None:
        print(f"Error: Table is not in a valid format")
        return None
    else:
        try:
            agent = create_pandas_dataframe_agent(
                ChatOpenAI(temperature=0, model="gpt-4.1-mini"),
                df,
                verbose=True,
                agent_type=AgentType.OPENAI_FUNCTIONS,
                allow_dangerous_code=True
            )
            result = agent.invoke({"input": query})
            return str(result)
        except Exception as e:
            return f"Error analysing table: {str(e)}"
    
