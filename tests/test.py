import multiprocessing
import unittest
from src.jarvis_conversationalist.logger_config import get_logger
from src.jarvis_conversationalist import process_assistant_response, get_core_path
from src.jarvis_conversationalist.audio_player import play_audio_file
from src.jarvis_conversationalist.openai_functions.functions import get_function_info


class TestJarvisConversationalist(unittest.TestCase):

    def test_get_logger(self):
        # Define a test case
        logger = get_logger()
        # Assert that the function returns the expected result
        self.assertIsNotNone(logger)

    def test_play_audio_file(self):
        # Define a test case
        file_path = get_core_path() + "/searching.wav"
        blocking = True
        play_audio_file(file_path, blocking)
        self.assertIsNotNone(blocking)

    def test_process_assistant_response(self):
        # Define a test case
        a = multiprocessing.Event()
        b = multiprocessing.Event()
        query = "What's the weather in Baltimore?"
        context = process_assistant_response(query, a, b)
        # Assert that the function returns the expected result
        self.assertIsNotNone(context)


# if __name__ == "__main__":
#     unittest.main()
