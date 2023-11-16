import time
import os
import re
from statistics import mean
import multiprocessing
import certifi
from numpy import average
import atexit
import importlib.resources as pkg_resources

from .openai_utility_functions import check_for_directed_at_me, check_for_completion, extract_query
from .openai_interface import stream_response, resolve_response, use_tools, schedule_refresh_assistant
from .streaming_response_audio import stream_audio_response, set_rt_text_queue
from .audio_player import play_audio_file
from .audio_listener import audio_capture_process
from .audio_transcriber import audio_processing_process


from .logger_config import get_logger
logger = get_logger()


try:
    core_path = str(pkg_resources.files('jarvis_conversationalist').joinpath('audio_files'))
except Exception as e:
    if os.getcwd().find('jarvis-conversationalist') != -1:
        core_path = os.path.join(os.getcwd()[:os.getcwd().find('jarvis-conversationalist')],
                                 'jarvis-conversationalist', 'src', 'jarvis_conversationalist', 'audio_files')
    else:
        raise Exception(f"Failed to find the audio files directory: {e}")

os.environ["REQUESTS_CA_BUNDLE"] = certifi.where()
os.environ["SSL_CERT_FILE"] = certifi.where()

wake_word = "jarvis"


def get_core_path():
    """
    Get the path to the core audio files.
    :return: The path to the core audio files.
    :rtype: str
    """
    return core_path


def process_assistant_response(query, beeps_stop_event, interrupt_event):
    """
    Processes the assistant's response to the user's query.
    :param query:
    :param beeps_stop_event:
    :param interrupt_event:
    :return: History of the conversation.
    :rtype: list
    """
    new_history = [{"content": query, "role": "user"}]
    current_stream = multiprocessing.Queue()
    set_rt_text_queue(current_stream)
    content, reason, tool_calls = stream_audio_response(stream_response(new_history),
                                                        stop_audio_event=beeps_stop_event,
                                                        skip=interrupt_event)
    if not interrupt_event.is_set():
        if tool_calls is None:
            new_history.append({"content": content, "role": "assistant"})
            logger.info("Response Content:"+" "+str(content))
        else:
            while tool_calls is not None and not interrupt_event.is_set():
                tool_stop_event = play_audio_file(core_path+"/searching.wav", loops=7, blocking=False)
                new_history += use_tools(tool_calls, content)
                tool_stop_event.set()
                content, reason, tool_calls = stream_audio_response(
                    stream_response(new_history, keep_last_history=True),
                    stop_audio_event=beeps_stop_event,
                    skip=interrupt_event)
            if tool_calls is None:
                new_history.append({"content": content, "role": "assistant"})
    if interrupt_event.is_set():
        logger.info("Interrupted")
        partial = ""
        while not current_stream.empty():
            partial += current_stream.get()["content"] + "\n"
        new_history.append({"content": partial, "role": "assistant"})
        new_history.append({"content": "Interupted by user.", "role": "system"})
    return new_history


def converse(memory, interrupt_event, process_queue):
    """
    Converse with the user.
    :param memory:
    :type memory: int
    :param interrupt_event:
    :type interrupt_event: multiprocessing.Event
    :param ready_event:
    :type ready_event: multiprocessing.Event
    :return: None
    """
    try:
        audio_queue = multiprocessing.Queue()
        text_queue = multiprocessing.Queue()

        speaking = multiprocessing.Event()
        capture_process = multiprocessing.Process(target=audio_capture_process,
                                                  args=(audio_queue, speaking), )
        processing_process = multiprocessing.Process(target=audio_processing_process,
                                                     args=(audio_queue, text_queue, speaking))
        atexit.register(capture_process.terminate)
        atexit.register(processing_process.terminate)

        capture_process.start()
        processing_process.start()

        # Rolling buffer to store text
        transcript = []
        timestamps = []
        schedule_refresh_assistant()
        try:
            test = audio_queue.get(timeout=10)
        except multiprocessing.queues.Empty:
            play_audio_file(core_path + "/tone_two.wav", blocking=False, loops=4)
            audio_queue.get()
        play_audio_file(core_path + "/tone_one.wav", blocking=False)
        process_queue.put("start")
        last_response_time = None
        while True:
            interrupt_event.clear()
            text = text_queue.get()
            if bool(re.search('[a-zA-Z0-9]', text)):
                transcript.append(text)
                timestamps.append(time.time())

            # Clean up old entries
            current_time = time.time()
            while transcript and timestamps[0] < current_time - memory:
                transcript.pop(0)
                timestamps.pop(0)
            transcript_long = len("\n".join(transcript)) > 25
            if last_response_time is None:
                time_since_last_response = 100000000
            else:
                time_since_last_response = time.time() - last_response_time
            # Check for "jarvis" and print buffer if found
            if wake_word in text.lower() or (time_since_last_response < 15 and transcript_long):
                logger.info(" - - Checking if what has been said was directed at me...")
                logger.info("\n".join(transcript))
                directed_at_results = check_for_directed_at_me(transcript)
                directed_at_results = [x for x in directed_at_results if x <= 1.0]
                if len(directed_at_results) == 0:
                    probably_at_me = False
                else:
                    target_intended = 0.8
                    probably_at_me = average(directed_at_results) > target_intended
                logger.info("Probability at me: " + str(directed_at_results))
                if probably_at_me:
                    completion_results = check_for_completion(transcript)
                    logger.info("Probability of completion:" + str(completion_results))
                    completion_results = [x for x in completion_results if x <= 1.0]
                    target_completion = 0.8
                    if len(completion_results) == 0:
                        completed = False
                    else:
                        completed = mean(completion_results) > target_completion
                    if not completed:
                        additions = 0
                        max_time = 25
                        max_additions = 3
                        current_time = time.time()
                        while time.time() - current_time < max_time and additions < max_additions:
                            if not text_queue.empty():
                                text = text_queue.get()
                                if bool(re.search('[a-zA-Z0-9]', text)):
                                    logger.info(" - - - Adding to transcript: " + text)
                                    transcript.append(text)
                                    timestamps.append(time.time())
                                    additions += 1
                                    completion_results = check_for_completion(transcript)
                                    logger.info("Still checking probability of completion:"+" "+str(completion_results))
                                    completion_results = [x for x in completion_results if x <= 1.0]
                                    if len(completion_results) == 0:
                                        completed = False
                                    else:
                                        completed = mean(completion_results) > target_completion
                                    if completed:
                                        additions = max_additions
                    speaking.set()
                    beeps_stop_event = play_audio_file(core_path+"/beeps.wav", loops=7, blocking=False)
                    extracted_query = extract_query(transcript)
                    logger.info("Query extracted: " + extracted_query)
                    if not interrupt_event.is_set():
                        try:
                            new_history = process_assistant_response(extracted_query, beeps_stop_event, interrupt_event)
                        except Exception as e:
                            logger.error(e)
                            play_audio_file(core_path + "/major_error.wav", blocking=False)
                            new_history = [{"content": extracted_query, "role": "user"},
                                           {"content": "I'm sorry, I'm having so issues with my circuits.",
                                            "role": "assistant"}]
                    logger.info("Resolving...")
                    resolve_response(new_history)
                    logger.info("Resolved")
                    if interrupt_event.is_set():
                        logger.info("Interrupted")
                        interrupt_event.clear()

                    transcript = []
                    timestamps = []
                    while not audio_queue.empty():
                        audio_queue.get()
                    while not text_queue.empty():
                        text_queue.get()
                    last_response_time = time.time()
                    speaking.clear()
                    play_audio_file(core_path+"/tone_one.wav", blocking=False, delay=2)

    except KeyboardInterrupt:
        return
