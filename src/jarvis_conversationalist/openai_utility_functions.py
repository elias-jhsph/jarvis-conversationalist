import json
import certifi
import os
from openai import OpenAI, _utils

client = OpenAI()
_utils._logs.logger.setLevel("CRITICAL")

os.environ["REQUESTS_CA_BUNDLE"] = certifi.where()
os.environ["SSL_CERT_FILE"] = certifi.where()

name = "Jarvis"


def check_for_directed_at_me(transcript, n=1):
    functions = [
        {
            "name": "configure_response",
            "description": "Configures the appropriate response to the user's input.",
            "parameters": {
                "type": "object",
                "properties": {
                    "probability": {
                        "type": "number",
                        "description": "from 0 to 100, the likelihood that the user was speaking directly to"
                                       " and not about "+name+". Remember that the user may be speaking to someone "
                                       "else and that should factor into the probability that the user is speaking "
                                       "to "+name+" as opposed to "
                                                  "about "+name+". For example, if the user says '"+name+", "
                                       "please turn on the lights', the probability should be high that the user is "
                                       "speaking to "+name+". However, if the user says 'I was talking to "+name+" "
                                       "and he turned on the lights', the probability should be low (63) that the user "
                                       "is speaking to "+name+". Or, if the user says 'John, please turn on the lights "
                                       "for me', the probability should be low (15) that the user is speaking"
                                                              " to "+name+"."
                    }
                },
                "required": ["probability"],
            },
        }
    ]
    system_message = "You are seeing a live transcription of what is being said in a room." \
                     " There are many reasons why names might be mentioned. There may also be multiple" \
                     " people in the room or people on the phone. It is your job to determine if the user is speaking" \
                     " to " + name + " directly."

    response = client.chat.completions.create(model="gpt-3.5-turbo",
                                              temperature=0.4,
                                              messages=[{"role": "system", "content": system_message},
                                                        {"role": "user", "content": "\n".join(transcript)}],
                                              functions=functions,
                                              n=n,
                                              function_call={"name": "configure_response"})
    probabilities = []
    for result in response.choices:
        result = result.message.function_call
        if result:
            probabilities.append(json.loads(result.arguments)['probability']/100)
    return probabilities


def check_for_completion(transcript, n=1):
    """
    What is the likelihood that the user is done speaking?

    :param transcript: an array of strings representing the user's speech
    :param n: the number of responses to generate
    :return: list of likelihoods that the user is done speaking
    """
    functions = [
        {
            "name": "configure_response",
            "description": "Configures the appropriate response to the user's input.",
            "parameters": {
                "type": "object",
                "properties": {
                    "probability": {
                        "type": "number",
                        "description": "A value from 0 to 100, the probability that the user is done speaking based on "
                                       "ALL of the following core principles: "
                                       "The user has completed their full thought. The user has "
                                       "completed their sentence. The user has included all components of their "
                                       "response that they intended to include at the beginning of their statement."
                                       "If the user has not completed their thought, sentence, or included all parts"
                                       " of their response, the probability should be low (below 10). "
                                       "If the user ended their input without punctuation, and 'and' or 'but', the "
                                       "probability should be low (below 10)."
                                       "If the user has completed some of their thought, sentence, or included some "
                                       "parts of their response, the probability should be medium (between 10 and 55)."
                                       "If the user has completed their thought, sentence, and included all parts of "
                                       "their response, the probability should be high (above 72)."
                    }
                },
                "required": ["probability"],
            },
        }
    ]
    system_message = "You are seeing a live transcription of what is being said in a room. It is your job to determine"
    " if the user is done speaking by analyzing the text below and seeing if the user has completed their thought."

    response = client.chat.completions.create(model="gpt-4",
                                              temperature=0.4,
                                              messages=[{"role": "system", "content": system_message},
                                                        {"role": "user", "content": "\n".join(transcript)}],
                                              functions=functions,
                                              n=n,
                                              function_call={"name": "configure_response"})
    probabilities = []
    for result in response.choices:
        result = result.message.function_call
        if result:
            probabilities.append(json.loads(result.arguments)["probability"]/100)
    return probabilities


def extract_query(transcript, speaker_detection=True):
    """
    Extracts the query from the user's speech.

    :param transcript: an array of strings representing the user's speech
    :return: the query
    """
    functions = [
        {
            "name": "configure_response",
            "description": "Configures the appropriate response to the user's input.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "The query that the user is asking. For example, if the user says 'What is the "
                                       "weather like today "+name+"?', the query should be "
                                       "'What is the weather like today, "+name+"?'. Or if the user says '"+name+", "
                                       "please turn on the lights', the query should be "
                                       "'"+name+", Please turn on the lights'."
                    }
                },
                "required": ["query"],
            },
        }
    ]
    system_message = "You are seeing a live transcription of what is being said in a room. It is your job to " \
                     "determine the query that the user is asking by analyzing the text below and extracting word " \
                     "for word the section of the transcript that is the query, include who is asking it using " \
                     "the bracket format like '[Unknown Speaker X]: ' or '[Jane]: ' or '[John Doe]: ' etc. " \
                     "Ignore the rest of the unrelated transcript. Keep in mind that sometimes context from " \
                     "earlier parts of the conversation are critical to understanding a query - make sure to include" \
                     "all the context needed to complete the query well. " \
                     "There may be multiple people in the room or people " \
                     "on the phone. It is your job to determine which part of the transcript is the query meant " \
                     "for " + name + ". The query should be a question or a command or a statement directed at or" \
                                     "highly related to " + name + ".  It is ok if there are multiple parts of the " \
                                                                   "query, or if multiple people appear to be asking " \
                                                                   "questions to" + name + ", it is ok to include " \
                                                                                           "all of those subsections " \
                                                                                           "in the query - just make " \
                                                                                           "sure to include the " \
                                                                                           "speaker annotation for " \
                                                                                           "each subsection."
    if not speaker_detection:
        system_message = "You are seeing a live transcription of what is being said in a room. It is your job to " \
                         "determine the query the user is asking by analyzing the text below and extracting word " \
                         "for word the section of the transcript that is the query. Ignore the rest of the unrelated " \
                         "transcript. Keep in mind that sometimes context from earlier parts of the conversation are " \
                         "critical to understanding a query - make sure to include all the context needed to complete" \
                         " the query well. There may be multiple people in the room or people on the phone. It's your" \
                         " job to determine which part of the transcript is the query. The query should be a question" \
                         " or a command or a statement directed at or highly related to " + \
                         name + ".  It is ok if there are multiple parts of the query, or if multiple people appear " \
                                "to be asking questions to" + name + ", it is ok to include all of those subsections " \
                                                                     "in the query - just make sure to include the " \
                                                                     "speaker annotation for each subsection."

    response = client.chat.completions.create(model="gpt-4",
                                              messages=[{"role": "system", "content": system_message},
                                                        {"role": "user", "content": "\n".join(transcript)}],
                                              functions=functions,
                                              function_call={"name": "configure_response"})
    result = response.choices[0].message.function_call
    if result:
        return json.loads(result.arguments)["query"]
    return


def simple_stream_response(query, context=None):
    if context is None:
        context = []
    return client.chat.completions.create(model="gpt-4",
                                          messages=context+[{"role": "user", "content": query}],
                                          stream=True)
