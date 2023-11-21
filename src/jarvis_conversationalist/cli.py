import argparse
import threading
import time

from .conversationalist import converse
from .config import load_key, load_config, save_config, set_openai_key, set_user, get_openai_key
from .logger_config import change_logging_level
import warnings

warnings.filterwarnings("ignore", message="DeprecationWarning: 'audioop' is deprecated and "
                                          "slated for removal in Python 3.13")


def main():
    parser = argparse.ArgumentParser(description='Jarvis Conversationalist CLI')
    parser.add_argument('--user', type=str, help='The username for the OpenAI API')
    parser.add_argument('--key', type=str, help='The key for the OpenAI API')
    parser.add_argument('--reset', action='store_true', help='Reset the saved username and key')
    parser.add_argument('--verbose', action='store_true', help='Set the logging level to INFO')
    args = parser.parse_args()

    key = load_key()
    config = load_config(key)

    if args.verbose:
        change_logging_level("INFO")
    else:
        change_logging_level("ERROR")

    if args.reset:
        config = {}
    else:
        if args.user:
            config['user'] = args.user
        if args.key:
            config['key'] = args.key

    save_config(config, key)

    set_openai_key(config.get('key', None))
    set_user(config.get('user', 'User'))

    if get_openai_key() is None:
        print("Please set your OpenAI API key using the --key argument once to cache your key.")
        return
    print("Please wait while Jarvis boots...", end='\r')
    interrupt_event = threading.Event()
    start_event = threading.Event()
    stop_event = threading.Event()
    conversation_thread = threading.Thread(target=converse, args=(60, interrupt_event, start_event, stop_event),)
    conversation_thread.start()

    hello = "\rSay my name, 'Jarvis', to get my attention... Click \"Enter\" to interrupt me. " \
            "Click \"Esc\" then \"Enter\" to turn me off.\033[K"
    LINE_UP = '\033[1A'
    LINE_CLEAR = '\x1b[2K'
    try:
        while True:
            if conversation_thread.is_alive():
                if not start_event.is_set():
                    print("\033[KI am booting...\033[K", end='\r')
                    start_event.wait(timeout=1)
                else:
                    user_input = input(hello)
                    if user_input.strip() == '' and not interrupt_event.is_set():
                        interrupt_event.set()
                        # Clear the line and then print "Interrupted."
                        print(end=LINE_CLEAR)
                        print(LINE_UP, end=LINE_CLEAR)
                        print("\033[KSorry for going on and on.\033[K", end='\r')
                        reset_time = time.time()
                    elif user_input == '\x1b':
                        # Clear the line and then print "Quitting Jarvis"
                        print("\r\033[KI am starting my shutdown process...\033[K", end='\r')
                        stop_event.set()
                        while conversation_thread.is_alive():
                            threading.Event().wait(timeout=10)
                        conversation_thread.join(timeout=10)
                        return
            else:
                raise Exception("Conversation process is not alive!")

    except KeyboardInterrupt:
        print()
        print("Shutting down ASAP! Please Wait...")
        stop_event.set()
        while conversation_thread.is_alive():
            print("Shutting down...")
            threading.Event().wait(timeout=10)
        conversation_thread.join(timeout=10)
        return


if __name__ == "__main__":
    main()

