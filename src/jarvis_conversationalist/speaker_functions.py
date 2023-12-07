from .audio_identifier import SpeakerIdentifier, db_speaker_id
from .logger_config import get_log_folder_path
from .config import get_speakers_active

dir_root = get_log_folder_path()
speaker_pipeline = SpeakerIdentifier(persist_directory=dir_root)
speakers_active = get_speakers_active()


def disable_speaker_functions():
    global speakers_active
    speakers_active = False


def get_speaker_system_appendix():
    out = "You can hear anyone in room speaking. That is why you have been given an annotation of who is speaking. " \
          "In the form of:\n'[Unknown Speaker X]:  ...' or '[John Doe]:  ...' or '[Jane]:  ...'\n Do not include " \
          "any speaker annotation or any other brackets in your response. It is your job to keep track of who is " \
          "speaking and to respond to the correct person. If you are unsure who is speaking, ask for clarification. " \
          "The only way the system can know who is speaking is if you tell it to associate an Unknown Speaker with a " \
          "name by asking the user(s) to provide a name and storing it with the correct function. The user(s) do not " \
          "know what 'Unknown Speaker X' number they are. Do not ask them to provide a name for a specific " \
          "'Unknown Speaker X' number. Instead, ask them to provide a name for the person that is speaking, " \
          "who spoke last, or who said something specific. If you are unsure who is speaking, ask for clarification. " \
          "This means you CAN recognize voices, just trust your annotations (remember the same person may talk with " \
          "multiple voices)."
    return out


def store_name_for_unknown_speaker(unknown_speaker_id="", name=""):
    if name == "":
        raise ValueError("The name cannot be empty.")
    if name.lower().find("unknown") != -1:
        raise ValueError("The name cannot contain 'Unknown'.")
    if name.lower().find("speaker") != -1:
        raise ValueError("The name cannot contain 'Speaker'.")
    if unknown_speaker_id == "":
        raise ValueError("The unknown_speaker_id cannot be empty.")
    if unknown_speaker_id.find("Unknown Speaker ") == -1:
        if unknown_speaker_id.isnumeric():
            unknown_speaker_id = "Unknown Speaker "+unknown_speaker_id
        else:
            raise ValueError("The unknown_speaker_id must start with 'Unknown Speaker '.")
    try:
        embedding = speaker_pipeline.get_unknown_embedding(unknown_speaker_id)
    except TypeError:
        embedding = None
    if embedding is None:
        raise ValueError("The unknown_speaker_id does not exist.")
    speaker_pipeline.add_known_speaker(embedding, name)
    speaker_pipeline.remove_unknown_speaker(unknown_speaker_id)
    return True


def store_name_for_unknown_speaker_documentation():
    schema = {"type": "function",
              "function": {
                    "name": "store_name_for_unknown_speaker",
                    "description": "Stores a name for an unknown speaker.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "unknown_speaker_id": {
                                "type": "string",
                                "description": "The unknown speaker id to store the name for ex. 'Unknown Speaker X' "
                                               "or just 'X'.",
                            },
                            "name": {
                                "type": "string",
                                "description": "The name to store for the unknown speaker.",
                            },
                        },
                        "required": ["unknown_speaker_id", "name"],
                    },
                }
            }
    examples = 'Examples:\n {"function_name": "store_name_for_unknown_speaker", "parameters": {"unknown_speaker_id": ' \
               '"Unknown Speaker 1", "name": "John Doe"}}\n{"function_name": "store_name_for_unknown_speaker", ' \
               '"parameters": {"unknown_speaker_id": "Unknown Speaker 2", "name": "Jane"}}\n'
    return schema, examples


def remove_name_of_known_speaker(name=""):
    if name == "":
        raise ValueError("The name cannot be empty.")
    if name.lower().find("unknown") != -1:
        raise ValueError("The name cannot contain 'Unknown'.")
    speaker_pipeline.remove_known_speaker(name)
    return True


def remove_name_of_known_speaker_documentation():
    schema = {"type": "function",
              "function": {
                    "name": "remove_name_of_known_speaker",
                    "description": "Removes a name of a known speaker.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "name": {
                                "type": "string",
                                "description": "The name of the known speaker to remove.",
                            },
                        },
                        "required": ["name"],
                    },
                }
            }
    examples = 'Examples:\n {"function_name": "remove_name_of_known_speaker", "parameters": {"name": "John Doe"}}\n' \
               '{"function_name": "remove_name_of_known_speaker", "parameters": {"name": "Jane"}}\n'
    return schema, examples


def get_speaker_function_info():
    if not speakers_active:
        return {}
    return {"store_name_for_unknown_speaker": {"function": store_name_for_unknown_speaker,
                                               "schema": store_name_for_unknown_speaker_documentation()[0],
                                               "examples": store_name_for_unknown_speaker_documentation()[1]},
            "remove_name_of_known_speaker": {"function": remove_name_of_known_speaker,
                                             "schema": remove_name_of_known_speaker_documentation()[0],
                                             "examples": remove_name_of_known_speaker_documentation()[1]},
            }


info = get_speaker_function_info()
function_list = []
function_examples = []
for val in info.values():
    function_list.append(val["schema"])
    function_examples.append(val["examples"])


def get_speaker_function_list():
    if not speakers_active:
        return []
    return function_list


def get_speaker_function_examples():
    if not speakers_active:
        return []
    return function_examples
