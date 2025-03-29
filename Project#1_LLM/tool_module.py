import os
import time
import requests
import asyncio
import logging
from typing import Any, List

from newspaper import Article
from duckduckgo_search import DDGS
import tiktoken

class ToolModule:
    """
    A collection of utility methods that the chatbot can call:
      - Web search (using DDGS from duckduckgo_search)
      - File read/write/alter
      - Newspaper article extraction
      - Token counting and text chunking
    """
    def __init__(self, search_timeout: int = 5, concurrency: int = 5) -> None:
        self.search_timeout = search_timeout
        self.concurrency = concurrency

    def search_web(self, query: str, max_results: int = 5) -> List[Any]:
        """Performs a synchronous web search using DuckDuckGo's DDGS."""
        try:
            logging.info(f"🔍 Searching the web for: {query}")
            results_list = []
            with DDGS() as ddgs:
                for result in ddgs.text(keywords=query, region="us-en", safesearch="Off"):
                    results_list.append({
                        "title": result.get("title", ""),
                        "href": result.get("href", "")
                    })
                    if len(results_list) >= max_results:
                        break
            logging.info(f"🔎 Search results: {results_list}")
            time.sleep(1)
            return results_list if results_list else [{"error": "No results found"}]
        except Exception as e:
            logging.error(f"Error in search_web: {e}", exc_info=True)
            return [{"error": str(e)}]

    async def search_web_async(self, query: str, max_results: int = 5) -> List[Any]:
        """Async version of the web search."""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, lambda: self.search_web(query, max_results))

    def read_file(self, file_path: str, mode: str = 'r', encoding: str = 'utf-8') -> str:
        """Read the contents of a file from the specified path."""
        try:
            with open(file_path, mode, encoding=encoding) as f:
                content = f.read()
                logging.info(f"Successfully read file: {file_path}")
                return content
        except FileNotFoundError:
            logging.error(f"File not found: {file_path}")
            return f"Error: File '{file_path}' not found."
        except PermissionError as e:
            logging.error(f"Permission denied when trying to read: {file_path}")
            return f"Error: Permission denied for file '{file_path}': {str(e)}"
        except UnicodeDecodeError as e:
            logging.error(f"Error decoding file '{file_path}' with encoding {encoding}")
            return f"Error: Unable to decode file '{file_path}' using {encoding}"
        except Exception as e:
            logging.error(f"Error reading file '{file_path}': {str(e)}", exc_info=True)
            return f"Error reading file '{file_path}': {str(e)}"

    def write_file(self, file_path: str, content: str, mode: str = 'w', encoding: str = 'utf-8') -> str:
        """Write the provided content to the specified file."""
        try:
            with open(file_path, mode, encoding=encoding) as f:
                f.write(content)
            logging.info(f"Successfully wrote to file: {file_path}")
            return content
        except FileNotFoundError as e:
            logging.error(f"File not found: {file_path}")
            return f"Error: File '{file_path}' not found."
        except PermissionError as e:
            logging.error(f"Permission denied when trying to write to: {file_path}")
            return f"Error: Permission denied for file '{file_path}': {str(e)}"
        except UnicodeEncodeError as e:
            logging.error(f"Error encoding content for file '{file_path}' with encoding {encoding}")
            return f"Error: Unable to encode content using {encoding}"
        except Exception as e:
            logging.error(f"Error writing to file '{file_path}': {str(e)}", exc_info=True)
            return f"Error writing to file '{file_path}': {str(e)}"

    def alter_file(self, file_path: str, search_text: str, replace_text: str) -> None:
        """Reads the file, replaces occurrences of search_text with replace_text, and writes it back."""
        if not os.path.isfile(file_path):
            logging.warning(f"alter_file: File not found: {file_path}")
            return
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            new_content = content.replace(search_text, replace_text)
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(new_content)
            logging.info(f"Replaced '{search_text}' with '{replace_text}' in {file_path}.")
        except Exception as e:
            logging.error(f"Error altering file {file_path}: {e}", exc_info=True)

    def extract_article(self, url: str) -> str:
        """Downloads and parses a news article from a given URL."""
        try:
            article = Article(url)
            article.download()
            article.parse()
            return article.text
        except Exception as e:
            logging.error(f"Error extracting article from {url}: {e}", exc_info=True)
            return f"Error extracting article: {e}"

    def count_tokens(self, text: str, encoding_name: str = "cl100k_base") -> int:
        """Counts the number of tokens in a given text using tiktoken."""
        try:
            encoding = tiktoken.get_encoding(encoding_name)
            tokens = encoding.encode(text)
            return len(tokens)
        except Exception as e:
            logging.error(f"Error counting tokens: {e}", exc_info=True)
            return 0

    def chunk_text(self, text: str, max_tokens: int = 1024, encoding_name: str = "cl100k_base") -> List[str]:
        """Splits text into chunks each up to max_tokens tokens in length."""
        try:
            encoding = tiktoken.get_encoding(encoding_name)
            all_tokens = encoding.encode(text)
            chunks = []
            start = 0
            while start < len(all_tokens):
                end = start + max_tokens
                chunk_tokens = all_tokens[start:end]
                chunk_text = encoding.decode(chunk_tokens)
                chunks.append(chunk_text)
                start = end
            return chunks
        except Exception as e:
            logging.error(f"Error chunking text: {e}", exc_info=True)
            return [text]

