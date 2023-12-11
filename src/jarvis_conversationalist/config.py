import os
import sys
import json
from cryptography.fernet import Fernet

user_home = os.path.expanduser("~")
logs_dir = os.path.join(user_home, "Jarvis Logs")
if not os.path.exists(logs_dir):
    os.mkdir(logs_dir)

CONFIG_FILE = os.path.join(logs_dir, 'config.json')
KEY_FILE = os.path.join(logs_dir, 'key.key')
USER = "User"
KEY = None
file = {}
if sys.platform != "darwin":
    SPEAKERS = True
else:
    SPEAKERS = False


def load_key():
    if os.path.exists(KEY_FILE):
        with open(KEY_FILE, 'rb') as f:
            return f.read()
    else:
        key = Fernet.generate_key()
        with open(KEY_FILE, 'wb') as f:
            f.write(key)
        return key


def load_config(key):
    cipher_suite = Fernet(key)
    if os.path.exists(CONFIG_FILE):
        with open(CONFIG_FILE, 'rb') as f:
            encrypted_data = f.read()
            decrypted_data = cipher_suite.decrypt(encrypted_data)
            return json.loads(decrypted_data)
    else:
        return {}


def save_config(config, key):
    cipher_suite = Fernet(key)
    with open(CONFIG_FILE, 'wb') as f:
        encrypted_data = cipher_suite.encrypt(json.dumps(config).encode())
        f.write(encrypted_data)


try:
    file = load_config(load_key())
except Exception as e:
    pass
if "user" in file:
    USER = file["user"]
if "key" in file:
    KEY = file["key"]
if "speakers" in file:
    SPEAKERS = file["speakers"]


def set_user(user):
    global USER
    USER = user


def set_openai_key(key):
    global KEY
    KEY = key


def set_speakers_active(speakers):
    global SPEAKERS
    SPEAKERS = speakers


def get_user():
    return USER


def get_openai_key():
    return KEY


def get_speakers_active():
    return SPEAKERS
