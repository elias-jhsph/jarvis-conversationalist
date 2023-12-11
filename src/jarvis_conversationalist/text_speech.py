import subprocess
import sys
import os
import uuid
import io
import re

if sys.platform != "darwin" and sys.platform != "linux":
    from pydub import AudioSegment
    from gtts import gTTS

if not os.path.exists(os.path.join(os.path.expanduser("~"), "Jarvis Logs")):
    os.mkdir(os.path.join(os.path.expanduser("~"), "Jarvis Logs"))
audio_folder = os.path.join(os.path.expanduser("~"), "Jarvis Logs", "temp_audio")
if not os.path.exists(audio_folder):
    os.mkdir(audio_folder)

if sys.platform == 'darwin':
    out = subprocess.run(['say', '-v', '?'], capture_output=True)
    vflag = []
    if out.stdout.decode("utf-8").find("Samantha (Enhanced)") >= 0:
        vflag = ['-v', 'Samantha (Enhanced)']
    if out.stdout.decode("utf-8").find("Tom (Enhanced)") >= 0:
        vflag = ['-v', 'Tom (Enhanced)']
    if out.stdout.decode("utf-8").find("Evan (Enhanced)") >= 0:
        vflag = ['-v', 'Evan (Enhanced)']
if sys.platform == 'linux':
    env = os.environ.copy()
    env["PATH"] = sys.executable + os.pathsep + env["PATH"]


class TextToSpeechError(Exception):
    """
    Exception raised when the text to speech conversion fails.
    """
    def __init__(self, sentence):
        """
        Initialise the exception.

        :param sentence: str, The sentence that failed to be converted to speech.
        :type sentence: str
        """
        self.sentence = sentence
        super().__init__(f"The following sentence was too long to turn into voice '{self.sentence}'.")


def simplify_urls(text):
    """
    Simplify URLs in the given text by removing the protocol and "www." and anything after the domain name.
    :param text: The text to simplify URLs in.
    :type text: str
    :return: The modified text.
    :rtype: str
    """
    # Regular expression to match URLs
    url_pattern = r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'

    # Find all URLs in the text
    urls = re.findall(url_pattern, text)

    # Function to extract and simplify the domain name from a URL
    def get_simplified_domain(url):
        domain = re.sub(r'(http[s]?://|www\.)', '', url)  # Remove the protocol and "www."
        domain = re.sub(r'(/.*)', '', domain)            # Remove anything after the domain
        return domain

    # Replace each URL with its simplified domain name
    for url in urls:
        simplified_domain = get_simplified_domain(url)
        text = text.replace(url, simplified_domain)

    return text


def find_longest_sentence(text):
    """
    Find the longest sentence in the given text.
    :param text: The text to find the longest sentence in.
    :type text: str
    :return: The longest sentence.
    :rtype: str
    """
    # Split the text into sentences using regex
    sentences = re.split(r' *[\.\?!][\'"\)\]]* *', text)

    # Find the longest sentence
    longest_sentence = max(sentences, key=len)

    return longest_sentence


def split_longest_sentence(text):
    """
    Split the longest sentence in the given text into smaller sentences.
    :param text: The text to split the longest sentence in.
    :type text: str
    :return: The modified text.
    :rtype: str
    """
    # Find the longest sentence
    longest_sentence = find_longest_sentence(text)

    # Split the longest sentence into chunks
    chunks = split_sentence(longest_sentence)

    # Replace the longest sentence in the text with the smaller sentences
    new_longest_sentence = '. '.join(chunks)
    modified_text = text.replace(longest_sentence, new_longest_sentence)

    return modified_text


def capitalize_first_letter(sentence: str) -> str:
    """
    Capitalize the first letter of the given sentence.

    :param sentence: The sentence to capitalize the first letter.
    :type sentence: str
    :return: The sentence with the first letter capitalized.
    :rtype: str
    """
    if len(sentence) > 0:
        return sentence[0].upper() + sentence[1:]
    return sentence


def remove_trailing_comma(sentence: str) -> str:
    """
    Remove trailing comma from the given sentence.

    :param sentence: The sentence to remove the trailing comma.
    :type sentence: str
    :return: The sentence without the trailing comma.
    :rtype: str
    """
    return sentence.rstrip(',')


def find_best_split(sentence: str) -> int:
    """
    Find the best split index for a given sentence, considering commas, colons, and semicolons.

    :param sentence: The input sentence to find the best split index.
    :return: The best split index or None if there are no delimiters.
    :rtype: int or None
    """
    # Find the indices of all delimiters
    delimiter_indices = [m.start() for m in re.finditer(r'[:,;]', sentence)]

    if not delimiter_indices:
        return None

    # Calculate the lengths of the two halves for each delimiter
    half_lengths = [(abs(len(sentence[:i]) - len(sentence[i+1:])), i) for i in delimiter_indices]

    # Find the delimiter that results in the most evenly-sized halves
    best_split_index = min(half_lengths, key=lambda x: x[0])[1]

    return best_split_index


def split_sentence(sentence: str) -> list:
    """
    Split the input sentence into two parts at the best delimiter.

    :param sentence: The input sentence to split.
    :type sentence: str
    :return: A list of two sentences after splitting.
    :rtype: list
    """
    best_split_index = find_best_split(sentence)

    if best_split_index is None:
        return [sentence]

    first_half = sentence[:best_split_index].strip()
    second_half = sentence[best_split_index + 1:].strip()

    # Capitalize the first letter and remove trailing commas
    first_half = capitalize_first_letter(remove_trailing_comma(first_half))
    second_half = capitalize_first_letter(remove_trailing_comma(second_half))

    return [first_half, second_half]


def text_to_speech(text: str, model="gpt-4", stream=False):
    """
    Convert the given text to speech using the specified model.

    :param text: The text to convert to speech.
    :type text: str
    :param model: The model to use for text-to-speech.
    :type model: str
    :param stream: Whether to return the audio content as a stream.
    :type stream: bool
    :return: The path to the audio file or the audio content as a stream.
    :rtype: str or bytes
    """
    slow_flag = True
    if model.find("gpt-4") >= 0:
        slow_flag = False
    if sys.platform == 'darwin':
        fixed_text = text.replace('"', r'\"')
        pitch = "44"
        if slow_flag:
            pitch = "40"
        first_word = fixed_text.split(" ")[0]
        rest_of_text = fixed_text.replace(first_word, "")
        fixed_text = "[[rate 175]] " + first_word + "[[rate 200]] " + rest_of_text
        text_cmd = f'[[pbas {pitch}]] [[slnc 100]]{fixed_text}[[slnc 100]]'
        output_file = os.path.join(audio_folder, str(uuid.uuid4()) + ".wav")
        result = subprocess.run(['say']+vflag+[text_cmd, "-o", output_file, '--data-format=LEI16@22050'],
                                capture_output=True)
        if not stream:
            return output_file
        if result.returncode != 0:
            raise Exception("Say command error: " + result.stderr.decode("utf-8"))
        with open(output_file, 'rb') as file:
            file.seek(0)
            byte_data = file.read()
        os.remove(output_file)
        return byte_data
    if sys.platform == 'linux':
        fixed_text = text.replace('"', r'\"')
        speed = ".85"
        if slow_flag:
            speed = "1"
        text_cmd = fixed_text
        output_file = os.path.join(audio_folder, str(uuid.uuid4()) + ".wav")
        result = subprocess.run(f"mimic3 \"{text_cmd}\" --length-scale {speed} > '{output_file}'",
                                capture_output=True,
                                env=env, shell=True)
        if not stream:
            return output_file
        if result.returncode != 0:
            raise Exception("Say command error: " + result.stderr.decode("utf-8"))
        with open(output_file, 'rb') as file:
            file.seek(0)
            byte_data = file.read()
        os.remove(output_file)
        return byte_data
    else:
        tts = gTTS(text, lang='en', slow=slow_flag)

        if stream:
            mp3_buffer = io.BytesIO()
            tts.write_to_fp(mp3_buffer)
            mp3_buffer.seek(0)
            mp3_segment = AudioSegment.from_file(mp3_buffer, format="mp3")
            wav_buffer = io.BytesIO()
            mp3_segment.export(wav_buffer, format="wav")
            wav_buffer.seek(0)
            audio_content = wav_buffer.read()
            return audio_content
        else:
            output_file = os.path.join(audio_folder, str(uuid.uuid4()) + ".wav")
            mp3_output = io.BytesIO()
            tts.write_to_fp(mp3_output)
            mp3_output.seek(0)
            mp3_segment = AudioSegment.from_file(mp3_output, format="mp3")
            mp3_segment.export(output_file, format="wav")
            return output_file
