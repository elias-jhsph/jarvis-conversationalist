import argparse
import multiprocessing
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

    interrupt_event = multiprocessing.Event()
    process_queue = multiprocessing.Queue()

    # Start the conversation in a separate process
    conversation_process = multiprocessing.Process(target=converse, args=(60, interrupt_event, process_queue),)
    conversation_process.start()

    reset_time = None
    print()
    hello = "\rListening... Click \"Enter\" to interrupt Jarvis. Click \"Esc\" then \"Enter\" to Quit.\033[K"
    LINE_UP = '\033[1A'
    LINE_CLEAR = '\x1b[2K'
    try:
        while True:
            if conversation_process.is_alive():
                if process_queue.empty():
                    print("\033[KBooting...\033[K", end='\r')
                    try:
                        test = process_queue.get(timeout=1)
                        if test == "start":
                            print("\033[KBooted!\033[K", end='\r')
                            process_queue.put("start")
                    except multiprocessing.queues.Empty:
                        continue
                else:
                    if reset_time is not None:
                        if time.time() - reset_time > 2:
                            user_input = input(hello)
                            reset_time = None
                    else:
                        user_input = input(hello)
                    if user_input.strip() == '' and not interrupt_event.is_set():
                        interrupt_event.set()
                        # Clear the line and then print "Interrupted."
                        print(end=LINE_CLEAR)
                        print(LINE_UP, end=LINE_CLEAR)
                        print("\033[KInterrupted.\033[K", end='\r')
                        reset_time = time.time()
                    elif user_input == '\x1b':
                        # Clear the line and then print "Quitting Jarvis"
                        print("\r\033[KQuitting Jarvis.\033[K", end='\r')
                        raise KeyboardInterrupt
                        return
            else:
                raise Exception("Conversation process is not alive! Error code: "+str(conversation_process.exitcode))

    except KeyboardInterrupt:
        print()
        print("Interrupted by user!")
        conversation_process.join(timeout=10)
        return
    finally:
        # Ensure the conversation process is terminated
        conversation_process.join(timeout=10)


if __name__ == "__main__":
    main()

