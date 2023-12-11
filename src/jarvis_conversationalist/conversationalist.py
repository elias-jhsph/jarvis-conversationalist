import multiprocessing
import queue
import threading
import time
import os
import re
from statistics import mean
import certifi
from numpy import average
import importlib.resources as pkg_resources

from .openai_utility_functions import check_for_directed_at_me, check_for_completion, extract_query
from .openai_interface import stream_response, resolve_response, use_tools, schedule_refresh_assistant, \
    get_speaker_detection
from .streaming_response_audio import stream_audio_response, set_rt_text_queue
from .audio_player import play_audio_file
from .audio_listener import audio_capture_process
from .audio_transcriber import audio_processing_thread

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
    current_stream = queue.Queue()
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


def converse(memory, interrupt_event, start_event, stop_event):
    """
    Converse with the user.
    :param memory:
    :type memory: int
    :param interrupt_event: An event to indicate that the user has interrupted the assistant.
    :type interrupt_event: threading.Event
    :param start_event: An event to indicate that the assistant has started.
    :type start_event: threading.Event
    :param stop_event: An event to indicate that the assistant should stop.
    :type stop_event: threading.Event
    :return:
    """
    audio_queue = multiprocessing.Queue()
    text_queue = multiprocessing.Queue()
    multiprocessing_stop_event = multiprocessing.Event()

    speaking = multiprocessing.Event()
    capture_process = multiprocessing.Process(target=audio_capture_process,
                                              args=(audio_queue, speaking, multiprocessing_stop_event), )
    text_process = multiprocessing.Process(target=audio_processing_thread,
                                           args=(audio_queue, text_queue, speaking, multiprocessing_stop_event), )

    capture_process.start()
    text_process.start()

    # Rolling buffer to store text
    transcript = []
    timestamps = []
    schedule_refresh_assistant()
    logger.info("Waiting for text queue...")
    while text_queue.empty() and not stop_event.is_set():
        threading.Event().wait(1)
    logger.info("Starting...")
    while not text_queue.empty() and not stop_event.is_set():
        try:
            text_queue.get(timeout=5)
        except multiprocessing.queues.Empty:
            pass
    play_audio_file(core_path + "/tone_one.wav", blocking=False)
    start_event.set()
    delays = []
    while not stop_event.is_set():
        interrupt_event.clear()
        try:
            text, ts = text_queue.get(timeout=1)
            text = text.strip()
            if text == "Thank you for watching." or text == "Thanks for watching!"\
                    or text == "Thanks for watching." or text == "Thank you for watching!"\
                    or text == "Thanks for watching!" or text == "You":
                text = ""
            if bool(re.search('[a-zA-Z0-9]', text)):
                delays.append(time.time() - ts)
                avg_delay = average(delays)
                if len(delays) > 10:
                    delays.pop(0)
                logger.info("Average delay:" + " " + str(avg_delay) + " Raw text: " + text)
                transcript.append(text)
                timestamps.append(time.time())
        except queue.Empty:
            text = ""
            # wait for 1 second
            threading.Event().wait(1)
        if text != "":
            # Clean up old entries
            current_time = time.time()
            while transcript and timestamps[0] < current_time - memory:
                transcript.pop(0)
                timestamps.pop(0)

            # Check for "jarvis" and print buffer if found
            if wake_word in text.lower():
                logger.info(" - - Checking if what has been said was directed at me...")
                logger.info("\n".join(transcript))
                directed_at_results = check_for_directed_at_me(transcript)
                directed_at_results = [x for x in directed_at_results if x <= 1.0]
                if len(directed_at_results) == 0:
                    probably_at_me = False
                else:
                    target_intended = 0.7
                    probably_at_me = average(directed_at_results) > target_intended
                logger.info("Probability at me: " + str(directed_at_results))
                if probably_at_me:
                    completion_results = check_for_completion(transcript)
                    logger.info("Probability of completion:" + str(completion_results))
                    completion_results = [x for x in completion_results if x <= 1.0]
                    target_completion = 0.7
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
                                text, ts = text_queue.get()
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
                            else:
                                threading.Event().wait(0.3)
                    speaking.set()
                    beeps_stop_event = play_audio_file(core_path+"/beeps.wav", loops=7, blocking=False)
                    extracted_query = extract_query(transcript, speaker_detection=get_speaker_detection())
                    logger.info("Query extracted: " + extracted_query)
                    new_history = None
                    if not interrupt_event.is_set():
                        try:
                            new_history = process_assistant_response(extracted_query, beeps_stop_event, interrupt_event)
                        except Exception as e:
                            logger.error(e)
                            play_audio_file(core_path + "/major_error.wav", blocking=False)
                            new_history = [{"content": extracted_query, "role": "user"},
                                           {"content": "I'm sorry, I'm having so issues with my circuits.",
                                            "role": "assistant"}]
                    if new_history:
                        logger.info("Resolving...")
                        resolve_response(new_history)
                        schedule_refresh_assistant()
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
                    speaking.clear()
                    play_audio_file(core_path+"/tone_one.wav", blocking=True)
    logger.info("Converse trying to shutdown")
    interrupt_event.set()
    speaking.set()
    multiprocessing_stop_event.set()
    audio_queue.put((None, time.time()))
    audio_queue.put((None, time.time()))
    audio_queue.put((None, time.time()))
    logger.info("Capture trying to shutdown")
    capture_process.join(timeout=10)
    if capture_process.is_alive():
        logger.warning("Terminating Capture...")
    capture_process.terminate()
    logger.info("Transcribe trying to shutdown")
    text_process.join(timeout=5)
    if text_process.is_alive():
        logger.warning("Terminating Transcribe...")
    capture_process.terminate()


if __name__ == "__main__":
    converse(60, threading.Event(), threading.Event(), threading.Event())
