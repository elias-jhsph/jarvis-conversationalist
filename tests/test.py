import multiprocessing
import threading
import unittest
import os
from src.jarvis_conversationalist.config import get_openai_key
if get_openai_key() is not None:
    os.environ["OPENAI_API_KEY"] = get_openai_key()
from src.jarvis_conversationalist.logger_config import get_logger
from src.jarvis_conversationalist.conversationalist import process_assistant_response, get_core_path, converse
from src.jarvis_conversationalist.audio_player import play_audio_file, shutdown_audio
from src.jarvis_conversationalist.openai_functions.functions import get_function_info


class TestJarvisConversationalist(unittest.TestCase):
    @classmethod
    def tearDownClass(cls):
        shutdown_audio()

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

    def test_converse_quit(self):
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
        start_event.wait(timeout=30)
        stop_event.set()
        print(threading.enumerate())
        conversation_thread.join(timeout=60)
        print(threading.enumerate())
        if not conversation_thread.is_alive():
            closed = True
        else:
            closed = False
        self.assertTrue(closed)

    # def test_process_assistant_response(self):
    #     # Define a test case
    #     beeps_stop_event = multiprocessing.Event()
    #     interrupt_event = threading.Event()
    #     query = "What's the weather in Baltimore?"
    #     context = process_assistant_response(query, beeps_stop_event, interrupt_event)
    #     # Assert that the function returns the expected result
    #     self.assertIsNotNone(context)
