# __main__.py
from .config import set_openai_key, get_openai_key
import argparse
import os
parser = argparse.ArgumentParser(description='Jarvis Conversationalist CLI')
parser.add_argument('--user', type=str, help='The username for the OpenAI API')
parser.add_argument('--key', type=str, help='The key for the OpenAI API')
parser.add_argument('--reset', action='store_true', help='Reset the saved username and key')
parser.add_argument('--verbose', action='store_true', help='Set the logging level to INFO')
args = parser.parse_args()

if get_openai_key() is None:
    if args.key:
        set_openai_key(args.key)
        os.environ["OPENAI_API_KEY"] = get_openai_key()
    else:
        print("Please set your OpenAI API key using the --key argument once to cache your key.")
else:
    os.environ["OPENAI_API_KEY"] = get_openai_key()

from .cli import main

if __name__ == "__main__":
    main()
