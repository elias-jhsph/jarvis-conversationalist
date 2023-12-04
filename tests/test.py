import logging
import multiprocessing
import threading
import unittest
import os
from datetime import datetime

from src.jarvis_conversationalist.config import get_openai_key
if get_openai_key() is not None:
    os.environ["OPENAI_API_KEY"] = get_openai_key()
from src.jarvis_conversationalist.logger_config import get_logger
from src.jarvis_conversationalist.conversationalist import process_assistant_response, get_core_path, converse
from src.jarvis_conversationalist.audio_player import play_audio_file
from src.jarvis_conversationalist.openai_functions.functions import get_function_info
from src.jarvis_conversationalist.audio_listener import audio_capture_process, listen_to_user, prep_mic


class TestJarvisConversationalist(unittest.TestCase):

    def test_get_logger(self):
        # Define a test case
        logger = get_logger()
        # Assert that the function returns the expected result
        self.assertIsNotNone(logger)

    def test_play_audio_file(self):
        # Define a test case
        file_path = get_core_path() + "/tone_two.wav"
        blocking = True
        play_audio_file(file_path, blocking)
        self.assertIsNotNone(blocking)

    # def test_listen_to_user(self):
    #     # Define a test case
    #     level = prep_mic()
    #     audio_data = listen_to_user(level)
    #     self.assertIsNotNone(audio_data)
    #
    # def test_audio_process(self):
    #     audio_queue = multiprocessing.Queue()
    #     speaking = multiprocessing.Event()
    #     multiprocessing_stop_event = multiprocessing.Event()
    #     capture_process = multiprocessing.Process(target=audio_capture_process,
    #                                               args=(audio_queue, speaking, multiprocessing_stop_event), )
    #     capture_process.start()
    #     self.assertTrue(capture_process.is_alive())
    #     # wait for audio queue to be populated
    #     audio_queue.get(timeout=60)

    def test_converse(self):
        # Define a test case
        timeout = 60
        interrupt_event = threading.Event()
        start_event = threading.Event()
        stop_event = threading.Event()
        conversation_thread = threading.Thread(target=converse, args=(timeout,
                                                                      interrupt_event,
                                                                      start_event,
                                                                      stop_event),)
        conversation_thread.start()
        self.assertIsNotNone(conversation_thread)
        get_logger().info(str(start_event.is_set())+" "+str(datetime.now()))
        start_event.wait(timeout=140)
        threading.Event().wait(timeout=5)
        get_logger().info(str(start_event.is_set()) + " " + str(datetime.now()))
        stop_event.set()
        threading.Event().wait(timeout=25)
        get_logger().info(threading.enumerate())
        conversation_thread.join(timeout=60)
        get_logger().info(threading.enumerate())
        if not conversation_thread.is_alive():
            closed = True
        else:
            closed = False
        self.assertTrue(closed)

    def test_process_assistant_response(self):
        # Define a test case
        beeps_stop_event = multiprocessing.Event()
        interrupt_event = threading.Event()
        query = "What's the weather in Baltimore?"
        context = process_assistant_response(query, beeps_stop_event, interrupt_event)
        # Assert that the function returns the expected result
        self.assertIsNotNone(context)
