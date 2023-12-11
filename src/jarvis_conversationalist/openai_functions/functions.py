import json

from .internet_helper import create_internet_context
from .internet_helper import search
from .weather_functions import get_weather
from .google_tools import simple_google_search
from .basic_tools import write_to_file, open_webpage, read_from_file, list_files


def deep_long_duration_internet_research_documentation():
    """
    Returns json documentation for the internet context function.

    :return: The json documentation for the internet context function.
    """
    schema = {"type": "function",
              "function": {
                "name": "deep_long_duration_internet_research",
                "description": "Searches the internet for a query and returns a detailed summary of the top "
                               "most relevant results.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "The query to search for.",
                        },
                    },
                    "required": ["query"],
                },
              }
              }
    examples = 'Examples:\n {"function_name": "search_helper", "parameters": {"query": "What is the current state' \
               'of the Israeli Palestinian Conflict"}}\n' \
               '{"function_name": "search_helper", "parameters": {"query": "What impactful new climate change ' \
               'research was released this month"}}\n'
    return schema, examples


def deep_long_duration_internet_research(query=None):
    return create_internet_context(query)


def search_weather(query=None):
    return get_weather(query)


def search_weather_documentation():
    """
    Returns json documentation for the get weather function.

    :return: The json documentation for the get weather function.
    """
    schema = {"type": "function",
              "function": {
                    "name": "search_weather",
                    "description": "Gets the weather for a location.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "query": {
                                "type": "string",
                                "description": "The location to get the weather for.",
                            },
                        },
                        "required": ["query"],
                    },
              }
              }
    examples = 'Examples:\n {"function_name": "search_weather", "parameters": {"query": "Paris France"}}\n ' \
               '{"function_name": "search_weather", "parameters": {"query": "Baltimore, MD"}}\n' \
               '{"function_name": "search_weather", "parameters": {"query": "New York City"}}\n'
    return schema, examples


def quick_google_answer(query=None):
    return simple_google_search(query)


def quick_google_answer_documentation():
    schema = {"type": "function",
                "function": {
                    "name": "quick_google_answer",
                    "description": "Searches google for query and returns simple processed answers"
                                   " if Google returns one",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "query": {
                                "type": "string",
                                "description": "The query to search for.",
                            },
                        },
                        "required": ["query"],
                    },
                }
                }
    examples = 'Examples:\n {"function_name": "quick_google_answer", "parameters": {"query": "What is population of ' \
               'france?"}}\n {"function_name": "quick_google_answer", "parameters": {"query": "What is the capital ' \
               'of France?"}}\n {"function_name": "quick_google_answer", "parameters": {"query": "What is the ' \
               'value of pi?"}}\n'
    return schema, examples


def get_top_google_results(query=None):
    return json.dumps(search(query, num_results=10, advanced=True)['items'])


def get_top_google_results_documentation():
    schema = {"type": "function",
                "function": {
                    "name": "get_top_google_results",
                    "description": "Searches google for query and returns the top 10 results",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "query": {
                                "type": "string",
                                "description": "The query to search for.",
                            },
                        },
                        "required": ["query"],
                    },
                }
                }
    examples = 'Examples:\n {"function_name": "get_top_google_results", "parameters": {"query": "Youtube video of how ' \
               'to make a cake"}}\n {"function_name": "get_top_google_results", "parameters": {"query": "Recipes to ' \
               'make a cake"}}'
    return schema, examples


def write_to_a_file(name=None, text=None):
    return write_to_file(name, text)


def write_to_a_file_documentation():
    schema = {"type": "function",
                "function": {
                    "name": "write_to_a_file",
                    "description": "Writes text to a file with the given name",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "name": {
                                "type": "string",
                                "description": "The name of the file to write to.",
                            },
                            "text": {
                                "type": "string",
                                "description": "The text to write to the file.",
                            },
                        },
                        "required": ["name", "text"],
                    },
                }
                }
    examples = 'Examples:\n {"function_name": "write_to_a_file", "parameters": {"name": "test.txt", "text": "This ' \
                  'is a test"}}\n {"function_name": "write_to_a_file", "parameters": {"name": "hello.json", "text": ' \
                    '"{\\"hello\\": \\"world\\"}"}}\n {"function_name": "write_to_a_file", "parameters": {"name": ' \
                    '"test.md", "text": "# This is a test"}}\n'
    return schema, examples


def read_from_a_file(name=None, folder=None):
    return read_from_file(name, folder)


def read_from_a_file_documentation():
    schema = {"type": "function",
                "function": {
                    "name": "read_from_a_file",
                    "description": "Reads text from a file with the given name",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "name": {
                                "type": "string",
                                "description": "The name of the file to read from.",
                            },
                            "folder": {
                                "type": "string",
                                "description": "The folder to read from. Must be one of: Documents, Downloads, or "
                                               "Desktop.",
                            },
                        },
                        "required": ["name", "folder"],
                    },
                }
                }
    examples = 'Examples:\n {"function_name": "read_from_a_file", "parameters": {"name": "test.txt", "folder": ' \
               '"Documents"}}\n {"function_name": "read_from_a_file", "parameters": {"name": "hello.json", "folder": ' \
               '"Downloads"}}\n {"function_name": "read_from_a_file", "parameters": {"name": "test.md", "folder": ' \
               '"Desktop"}}\n'
    return schema, examples


def list_files_in_folder(folder=None):
    return list_files(folder)


def list_files_in_folder_documentation():
    schema = {"type": "function",
                "function": {
                    "name": "list_files_in_folder",
                    "description": "Lists the files in the given folder",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "folder": {
                                "type": "string",
                                "description": "The folder to list files from. Must be one of: Documents, Downloads, "
                                               "or Desktop.",
                            },
                        },
                        "required": ["folder"],
                    },
                }
                }
    examples = 'Examples:\n {"function_name": "list_files_in_folder", "parameters": {"folder": "Documents"}}\n ' \
               '{"function_name": "list_files_in_folder", "parameters": {"folder": "Downloads"}}\n ' \
               '{"function_name": "list_files_in_folder", "parameters": {"folder": "Desktop"}}\n'
    return schema, examples


def open_url_for_user(url=None):
    return open_webpage(url)


def open_url_for_user_documentation():
    schema = {"type": "function",
                "function": {
                    "name": "open_url_for_user",
                    "description": "Opens a url in the user's default web browser.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "url": {
                                "type": "string",
                                "description": "The url to open.",
                            },
                        },
                        "required": ["url"],
                    },
                }
                }
    examples = 'Examples:\n {"function_name": "open_url_for_user", "parameters": {"url": "https://www.google.com"}}\n' \
               '{"function_name": "open_url_for_user", "parameters": {"url": "https://www.youtube.com"}}\n'
    return schema, examples


def get_function_info():
    return {"deep_long_duration_internet_research": {"function": deep_long_duration_internet_research,
                                                     "schema": deep_long_duration_internet_research_documentation()[0],
                                                     "examples": deep_long_duration_internet_research_documentation()[1]},
            "search_weather": {"function": search_weather,
                            "schema": search_weather_documentation()[0],
                            "examples": search_weather_documentation()[1]},
            "quick_google_answer": {"function": quick_google_answer,
                                    "schema": quick_google_answer_documentation()[0],
                                    "examples": quick_google_answer_documentation()[1]},
            "get_top_google_results": {"function": get_top_google_results,
                                        "schema": get_top_google_results_documentation()[0],
                                        "examples": get_top_google_results_documentation()[1]},
            "write_to_a_file": {"function": write_to_a_file,
                                "schema": write_to_a_file_documentation()[0],
                                "examples": write_to_a_file_documentation()[1]},
            "open_url_for_user": {"function": open_url_for_user,
                                    "schema": open_url_for_user_documentation()[0],
                                    "examples": open_url_for_user_documentation()[1]},
            "read_from_a_file": {"function": read_from_a_file,
                                    "schema": read_from_a_file_documentation()[0],
                                    "examples": read_from_a_file_documentation()[1]},
            "list_files_in_folder": {"function": list_files_in_folder,
                                        "schema": list_files_in_folder_documentation()[0],
                                        "examples": list_files_in_folder_documentation()[1]},
            }


info = get_function_info()
function_list = []
function_examples = []
for val in info.values():
    function_list.append(val["schema"])
    function_examples.append(val["examples"])


def get_function_list():
    return function_list


def get_function_examples():
    return function_examples


def get_system_appendix():
    base_message = "\n\nFunction Calling:\nDon't make assumptions about what values to plug into functions. " \
                   "Ask for clarification if a user request is ambiguous. Only use the functions in your " \
                   "function list. Make sure to always include all required parameters and to use the correct " \
                   "data types in the correct JSON format.\n"
    return base_message + "\n".join(function_examples)




