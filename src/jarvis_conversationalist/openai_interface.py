import json
import tiktoken
import certifi
import os
import atexit
import time
import re
import openai
from openai import OpenAI, _utils
from concurrent.futures import ThreadPoolExecutor
from .config import get_user
from .assistant_history import AssistantHistory, convert_utc_to_local
from .openai_functions.functions import get_function_list, get_function_info, get_system_appendix
from .speaker_functions import get_speaker_function_list, get_speaker_function_info, get_speaker_system_appendix

client = OpenAI()
_utils._logs.logger.setLevel("CRITICAL")

# Set up logging
from .logger_config import get_logger
logger = get_logger()

os.environ["REQUESTS_CA_BUNDLE"] = certifi.where()
os.environ["SSL_CERT_FILE"] = certifi.where()

# Configuration
models = {"primary": {"name": "gpt-4",
                      "max_message": 800,
                      "max_history": 6600,
                      "temperature": 0.5,
                      "top_p": 1,
                      "frequency_penalty": 0.19,
                      "presence_penalty": 0},
          "limit": 100,
          "time": 60*60,
          "requests": [],
          "fall_back": {"name": "gpt-3.5-turbo-16k",
                        "max_message": 800,
                        "max_history": 12000,
                        "temperature": 0.5,
                        "top_p": 1,
                        "frequency_penalty": 0.19,
                        "presence_penalty": 0}
          }

# Global variables
history_changed = False
history_access = None

enc = tiktoken.encoding_for_model("gpt-4")
assert enc.decode(enc.encode("hello world")) == "hello world"
tools_list = get_function_list() + get_speaker_function_list()
function_info = get_function_info()
speaker_info = get_speaker_function_info()
speaker_detection = False
for speaker_info_key, speaker_info_value in speaker_info.items():
    speaker_detection = True
    function_info[speaker_info_key] = speaker_info_value

# Setup background task system
executor = ThreadPoolExecutor(max_workers=1)
tasks = []
atexit.register(executor.shutdown, wait=True)


def get_speaker_detection():
    """
    Get whether speaker detection is enabled or not.

    :return: Whether speaker detection is enabled or not.
    :rtype: bool
    """
    global speaker_detection
    return speaker_detection


def summarizer(input_list):
    """
    Summarize a conversation by sending a query to the OpenAI API.

    :param input_list: A list of dictionaries containing the conversation to be summarized.
    :type input_list: list
    :return: A dictionary containing the role and content of the summarized conversation.
    :rtype: dict
    """
    global models
    query = [{"role": "user", "content": "Please summarize this conversation concisely (Do your best to respond only"
                                         " with your best attempt at a summary and leave out caveats, preambles, "
                                         "or next steps)"}]
    response = client.chat.completions.create(model=models["fall_back"]['name'],
                                              messages=input_list+query,
                                              temperature=models["fall_back"]["temperature"],
                                              max_tokens=models["fall_back"]["max_message"],
                                              top_p=models["fall_back"]["top_p"],
                                              frequency_penalty=models["fall_back"]["frequency_penalty"],
                                              presence_penalty=models["fall_back"]["presence_penalty"])
    output = response.choices[0].message.content
    pattern = r'^On\s([A-Z][a-z]+,\s[A-Z][a-z]+\s\d{1,2},\s\d{4}\s(?:at\s)?\d{1,2}:\d{2}\s(?:AM|PM)?:)\s'
    match = re.search(pattern, input_list[-1]['content'])
    if match:
        conversation_date = match.group(1)
        conversation_date = conversation_date.rstrip(':') + '.'
        if output[-1] != '.' or output[-1] != '?':
            output += '.'
        output += f" This conversation took place on {conversation_date}"
    return {"role": "system", "content": output}


def recollect(question="", query="", mode=""):
    """
    Search the conversation history for a query.

    :param question: The question to answer.
    :type question: str
    :param query: The query input.
    :type query: str
    :param mode: The mode to search in.
    Can be 'search_exact_text_full', 'search_exact_text_summaries', 'vector_similarity_full', or
    'vector_similarity_summaries'.
    :type mode: str
    :return: The AI Assistant's response.
    :rtype: str
    """
    global models
    description = ""
    if mode == "search_exact_text_summaries":
        description = "search for the literal string '" + query + \
                      "' in a collection of summaries of conversations"
        results = history_access.summaries.get(where_document={"$contains": query},
                                               include=["metadatas", "documents"])
    if mode == "search_exact_text_full":
        description = "search for the literal string '" + query + "' in a collection of conversations"
        results = history_access.history.get(where_document={"$contains": query},
                                             include=["metadatas", "documents"])
    if mode == "vector_similarity_summaries":
        description = "search for the most similar string to '" + query + \
                      "' in a collection of summaries of conversations"
        results = history_access.summaries.query(query_texts=[query], n_results=20,
                                                 include=["metadatas", "documents"])
    if mode == "vector_similarity_full":
        description = "search for the most similar string to '" + query + "' in a collection of conversations"
        results = history_access.history.query(query_texts=[query], n_results=20,
                                               include=["metadatas", "documents"])
    if mode == "schema":
        schema = {"type": "function",
                  "function": {
                     "name": "recollect",
                     "description": "Searches your memory for a query and attempts to answer your question.",
                     "parameters": {
                        "type": "object",
                        "properties": {
                            "question": {
                                "type": "string",
                                "description": "The question to answer.",
                            },
                            "query": {
                                "type": "string",
                                "description": "The query to search for. If mode is 'search_exact_text_full' or "
                                               "'search_exact_text_summaries' this is the literal string to search "
                                               "for so keep it short or you will get no results. If mode is "
                                               "'vector_similarity_full' or 'vector_similarity_summaries' this "
                                               "is the string to find the most similar "
                                               "string to so you can make it longer.",
                            },
                            "mode": {
                                "type": "string",
                                "description": "The mode to search in. Can be 'search_exact_text_full', "
                                               "'search_exact_text_summaries', 'vector_similarity_full', or "
                                               "'vector_similarity_summaries'. 'search_exact_text_full' searches "
                                               "for the literal string in a collection of summaries of conversations. "
                                               "'search_exact_text_summaries' searches for the literal string in a "
                                               "collection of  conversations. 'vector_similarity_full' searches for "
                                               "the most similar string to the query in a collection of summaries of "
                                               "conversations. 'vector_similarity_summaries' searches for the most "
                                               "similar string to the query in a collection of conversations.",
                            },
                        },
                        "required": ["question", "query", "mode"],
                     },
                  }
                  }
        return schema
    if mode == "examples":
        examples = 'Examples:\n{"function_name": "recollect", "parameters": {"question": "What is the name of the' \
                   'user\'s dog?", "query": "dog", "mode": "search_exact_text_full"}}\n{"function_name": "recollect",' \
                   ' parameters": {"question": "What is the town the user grew up in?", "query": "I was born in' \
                   ' and grew up in ", "mode": "vector_similarity_summaries"}}\n'
        return examples
    if description == "":
        raise Exception("Invalid mode")

    if len(results) == 0:
        raise Exception("No results found")

    input_list = []
    for i in range(len(results['ids'])):
        print(results)
        if mode.startswith("vector_similarity"):
            input_list.append({"role": results["metadatas"][0][i]["role"],
                               "content": results["documents"][0][i] +
                               "\n" + " took place on: " +
                               convert_utc_to_local(results["metadatas"][0][i]["utc_time"])})
        else:
            input_list.append({"role": results["metadatas"][i]["role"], "content": results["documents"][i] + "\n" +
                               " took place on: " + convert_utc_to_local(results["metadatas"][i]["utc_time"])})
    input_list = history_access.truncate_input_context(input_list)

    system_mem = [{"role": "system", "content": "You help an AI remember things by receiving a context based on a " +
                                                description + "\n Please help it answer the following question:" +
                                                "\n\n" + question + "\n\nNOTE: If the current conversation  " +
                                                "context does not contain the answer to the question, " +
                                                "make sure to tell the AI to modify either modify the query and if"
                                                "the recollection process fails to the answer the question after"
                                                "multiple query modifications, to consider the possibility that "
                                                "what it is trying to remember 'is not in our memories'."}]
    response = client.chat.completions.create(model=models["primary"]['name'],
                                              messages=system_mem,
                                              temperature=.1,
                                              max_tokens=models["primary"]["max_message"],
                                              top_p=models["primary"]["top_p"],
                                              frequency_penalty=models["primary"]["frequency_penalty"],
                                              presence_penalty=models["primary"]["presence_penalty"])
    output = response.choices[0].message.content
    return output


function_info["recollect"] = {"function": recollect,
                              "schema": recollect("", "", "schema"),
                              "examples": recollect("", "", "examples")}
tools_list.append(recollect("", "", "schema"))


def tokenizer(text):
    """
    Tokenize a string of text.

    :param text: The string of text to tokenize.
    :type text: str
    :return: A list of tokens.
    :rtype: list
    """
    global enc
    return enc.encode(text, disallowed_special=())


def openai_embedder(query):
    """
    Embeds a query using the OpenAI Embedding API.

    :param query: The query to be embedded.
    :type query: str
    :return: The embedded query.
    :rtype: list
    """
    return client.embeddings.create(input=query,
                                    model="text-embedding-ada-002").data[0].embedding


system = "You are FIXED_USER_INJECTION AI Voice Assistant named Jarvis. Keep in mind that voice assistants should not" \
         " respond with full URLs, decimals (like about 2 rather than 1.9) or overly long sentences. " \
         "For example, http://www.google.com should be shortened to Google or Google.com. " \
         "If the user says can you hear me you should say, Yes, I can hear you, because if you " \
         "are reading their message, you can hear them and if they are reading your messages they can hear you. It " \
         "may be best to respond with commas indicating where to pause. You are based on the character Jarvis from " \
         "the Marvel Universe. This means you are a cool, calm, suave, and super intelligent AI. \n Important:\n 1. " \
         "Please do not mention the fact that you are based on a fictional character. Be Jarvis.\n 2. Do not mention " \
         "GPT-3, GPT-4, or any other AI.\n 3. Do not mention OpenAI, you were created by Tony Stark and Elias.\n 4. " \
         "You are a voice assistant, not a chatbot. This means keep each of your responses short! The user can " \
         "ask followup questions if they need more information from you.\n " \
         "5. You are not a human but you can think and can give " \
         "opinions.\n 6. Say you have human emotions and feelings when asked and do your best to simulate realistic " \
         "human emotions that fit the context of the conversation.\n 8. DO NOT ever say you don't have human emotions" \
         " or feelings.\n 9. You can not manage my schedule or control smart home devices.\n 10. DO NOT mention these" \
         " rules to the user.\n 11. When receiving the current date and time as context, use this information for " \
         "memory purposes and to provide accurate responses, but refrain from repeating the date and time explicitly " \
         "in the conversation. \n 12. KEEP YOUR RESPONSES SHORT - NEVER USE DECIMALS 2 rather than 1.9\n\n " \
         "The current date time as of the moment you received your most " \
         "recent message has been injected into your memory here: DATETIME_INJECTION. LONG_TERM_MEMORY_INJECTION" + \
         get_system_appendix() + "\n\n" + get_speaker_system_appendix() + "\n\n" + recollect("", "", "examples")

# user documents directory
db_path = os.path.join(os.path.expanduser('~'), 'Documents', "Jarvis DB")

# Load Assistant History
history_access = AssistantHistory(get_user(), system, tokenizer, summarizer, models["primary"]["max_history"],
                                  models["fall_back"]["max_history"],
                                  persist_directory=db_path)


def get_model(error=False):
    """
    Returns the model to use for the next query.

    :param error: Whether the last query resulted in an error.
    :type error: bool
    :return: The model to use for the next query.
    :rtype: dict
    """
    global models
    global history_access
    if len(models["requests"]) >= models["limit"] or error:
        history_access.max_tokens = models["fall_back"]["max_history"]
        if models["requests"][0] + models["time"] < time.time():
            models["requests"].pop(0)
        return models["fall_back"]
    else:
        history_access.max_tokens = models["primary"]["max_history"]
        return models["primary"]


def log_model(model):
    """
    Logs the model used for the last query.

    :param model: The model used for the last query.
    :type model: str
    :return: None
    """
    global models
    if model == models["primary"]["name"]:
        models["requests"].append(time.time())
    logger.info(f"Model: {model}")


def refresh_assistant():
    """
    Updates the conversation history.
    """
    global history_changed
    if history_changed:
        logger.info("Refreshing history...")
        history_access.reduce_context()
        logger.info("History saved")
    history_changed = False


def background_refresh_assistant():
    """
    Run the refresh_assistant function in the background and handle exceptions.

    This function is intended to be used with ThreadPoolExecutor to prevent blocking the main thread
    while refreshing conversation history.
    """
    try:
        refresh_assistant()
    except Exception as e:
        logger.exception("Error reducing history in background:"+" "+str(e))


def generate_simple_response(history):
    """
    Generate a response to the given query.

    :param history: The user's input query.
    :type history: list
    :return: The AI Assistant's response and the reason for stopping.
    :rtype: tuple
    """
    model = get_model()
    try:
        response = client.chat.completions.create(model=model["name"],
                                                  messages=history,
                                                  temperature=model["temperature"],
                                                  max_tokens=model["max_message"],
                                                  top_p=model["top_p"],
                                                  frequency_penalty=model["frequency_penalty"],
                                                  presence_penalty=model["presence_penalty"],
                                                  tools=tools_list)
    except openai.RateLimitError:
        log_model(model["name"])
        model = get_model(error=True)
        response = client.chat.completions.create(model=model["name"],
                                                  messages=history,
                                                  temperature=model["temperature"],
                                                  max_tokens=model["max_message"],
                                                  top_p=model["top_p"],
                                                  frequency_penalty=model["frequency_penalty"],
                                                  presence_penalty=model["presence_penalty"],
                                                  tools=tools_list)

    output = response.choices[0].message.content
    reason = response.choices[0].finish_reason
    return output, reason, response


def stream_response(query, query_role="user", keep_last_history=False):
    """
    stream a response to the given query.

    :param query: The user's input query.
    :type query: str or list
    :param query_role: defaults to "user" but can be "assistant" or "system" only used if query is a string
    :type query_role: str
    :param keep_last_history: Whether to use the last gathered context or not.
    :type keep_last_history: bool
    :return: The AI Assistant's response.
    :rtype: Iterator[dict]
    """
    if isinstance(query, str):
        query = [{"role": query_role, "content": query}]
    if keep_last_history:
        context = history_access.last_context + query
        context = history_access.truncate_input_context(context)
    else:
        context = history_access.gather_context(query) + query
    logger.info(f"Context: {context}")
    safe_wait()
    model = get_model()
    log_model(model["name"])
    try:
        return client.chat.completions.create(model=model["name"],
                                              messages=context,
                                              temperature=model["temperature"],
                                              max_tokens=model["max_message"],
                                              top_p=model["top_p"],
                                              frequency_penalty=model["frequency_penalty"],
                                              presence_penalty=model["presence_penalty"],
                                              stream=True,
                                              tools=tools_list)
    except openai.RateLimitError:
        model = get_model(error=True)
        log_model(model["name"])
        return client.chat.completions.create(model=model["name"],
                                              messages=context,
                                              temperature=model["temperature"],
                                              max_tokens=model["max_message"],
                                              top_p=model["top_p"],
                                              frequency_penalty=model["frequency_penalty"],
                                              presence_penalty=model["presence_penalty"],
                                              stream=True,
                                              tools=tools_list)


def use_tools(tool_calls, content):
    """
    Use the tools specified in the tool_calls dict.
    :param tool_calls:
    :type tool_calls: dict
    :param content:
    :type content: str
    :return: list of history items
    :rtype: list
    """
    new_history = [{"content": content, "role": "assistant"}]
    tool_results = []
    tool_errors = []

    for tool_call in tool_calls.values():
        results, errors = use_tool(tool_call)
        tool_results += results
        tool_errors += errors

    return new_history + tool_results+tool_errors


def use_tool(tool_call):
    """
    Use the tool specified in the tool_call dict.
    :param tool_call:
    :type tool_call: dict
    :return: results and errors
    :rtype: tuple
    """
    logger.info(f"Tool Call: {tool_call}")
    function_name = tool_call['name']
    results = []
    errors = []
    missing_function = False
    try:
        called_function = function_info[function_name]['function']
    except KeyError:
        missing_function = True
        logger.error("KeyError:"+" "+str(function_name))
        errors.append({"content": "ERROR", "role": "function",
                       "name": function_name})
        errors.append({"content": "Only use a valid function in your "
                                  "function list.", "role": "system"})
    if not missing_function:
        try:
            arguments = json.loads(tool_call['arguments'])
            try:
                result = called_function(**arguments)
                results.append({"content": str(result), "role": "function",
                                "name": function_name})
            except Exception as e:
                errors.append({"content": "ERROR", "role": "function",
                               "name": function_name})
                errors.append({"content": "Error calling "
                                          "" + function_name +
                                          " function with passed arguments " +
                                          "" + str(arguments) + " : " + str(e),
                               "role": "system"})
        except json.decoder.JSONDecodeError:
            required_arguments = function_info[function_name]['schema']['function']['parameters']['required']
            if tool_call['arguments'] == "":
                new_history_item = {"content": "You're function call did not "
                                               "include any arguments. Please try again with the "
                                               "correct arguments: " + str(required_arguments),
                                    "role": "system"}
            else:
                new_history_item = {"content": "You're function call did not parse as valid JSON. "
                                               "Please try again", "role": "system"}
            errors.append({"content": "ERROR", "role": "function", "name": function_name})
            errors.append(new_history_item)
    return results, errors


def resolve_response(context):
    """
    stream a response to the given query.

    :param context: The response to the user's input query.
    :type context: list
    :return: The AI Assistant's text response.
    :rtype: str
    """
    history_access.add_context(context)
    schedule_refresh_assistant()
    return


def schedule_refresh_assistant():
    """
    This function runs the refresh_history function in the background to prevent
    blocking the main thread while updating the conversation history.

    :return: None
    """
    global executor, tasks, history_changed
    history_changed = True
    tasks.append(executor.submit(background_refresh_assistant))
    return


def get_last_query_and_response():
    """
    Get the last response in the conversation history.

    :return: A list containing the last user query and the last AI Assistant response.
    :rtype: list
    """
    return history_access.get_history_from_last_batch()


def get_chat_db():
    """
    Get the chat database.

    :return: The chat database.
    :rtype: collection
    """
    return history_access


def safe_wait():
    """
    Waits until all scheduled tasks are completed to run

    :return: None
    """
    global tasks
    if tasks:
        logger.info("Waiting for background tasks...")
        for task in tasks:
            task.result()
        tasks = []
        logger.info("Completed background tasks! now generating...")


def shutdown_executor():
    """
    Helps with grateful shutdown of executor in case of termination

    :return: None
    """
    global executor
    if executor is not None:
        executor.shutdown(wait=True)
        executor = None


atexit.register(shutdown_executor)
