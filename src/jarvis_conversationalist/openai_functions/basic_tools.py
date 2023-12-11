import os
import requests
from .internet_helper import truncate_content


def write_to_file(name, content):
    basename_of_name = os.path.basename(name)
    user_documents_path = os.path.expanduser("~/Documents")
    path = os.path.join(user_documents_path, basename_of_name)
    # make sure the path is a .txt file, a .md file, or a .json file
    if not path.endswith(".txt") and not path.endswith(".md") and not path.endswith(".json"):
        raise ValueError("Path must be a .txt, .md, or .json file.")
    with open(path, "w") as f:
        f.write(content)
    os.system(f"open {path}")
    return f'Wrote to path:"{path}" the content.'


def read_from_file(name, folder):
    if folder != "Documents" and folder != "Downloads" and folder != "Desktop":
        raise ValueError("Folder must be one of: Documents, Downloads, or Desktop.")
    basename_of_name = os.path.basename(name)
    user_documents_path = os.path.expanduser("~/"+folder)
    path = os.path.join(user_documents_path, basename_of_name)
    # make sure the path is a .txt file, a .md file, or a .json file
    if not path.endswith(".txt") and not path.endswith(".md") and not path.endswith(".json"):
        raise ValueError("Path must be a .txt, .md, or .json file.")
    with open(path, "r") as f:
        content = f.read()
    content = str(content)
    content = truncate_content(content)
    return content


def list_files(folder):
    if folder != "Documents" and folder != "Downloads" and folder != "Desktop":
        raise ValueError("Folder must be one of: Documents, Downloads, or Desktop.")
    path = os.path.expanduser("~/"+folder)
    # list .txt files, .md files, and .json files
    files = []
    for file in os.listdir(path):
        if file.endswith(".txt") or file.endswith(".md") or file.endswith(".json"):
            files.append(file)
    return files


def open_webpage(url):
    # see if the url is valid by calling the requests library
    r = requests.get(url)
    os.system(f"open {url}")
    return f"Opened {url} in browser."
