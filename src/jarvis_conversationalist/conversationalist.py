import sys
import time
import os
import re
from statistics import mean
import multiprocessing
import certifi
import wave
import audioop
from io import BytesIO
from soundfile import read
from numpy import float32
from numpy import frombuffer, int16, average
from pyaudio import PyAudio, paInt16, get_sample_size
import whisper
import warnings
import signal

from .openai_utility_functions import check_for_directed_at_me, check_for_completion, extract_query
from .openai_interface import stream_response, resolve_response, use_tools, schedule_refresh_assistant
from .streaming_response_audio import stream_audio_response
from .audio_player import play_audio_file


from .logger_config import get_logger
logger = get_logger()

warnings.filterwarnings("ignore", message="FP16 is not supported on CPU; using FP32 instead")

os.environ["REQUESTS_CA_BUNDLE"] = certifi.where()
os.environ["SSL_CERT_FILE"] = certifi.where()

# Initialize the speech recognition module and microphone
paudio = PyAudio()
device_info = paudio.get_default_input_device_info()
paudio.terminate()
default_sample_rate = int(device_info["defaultSampleRate"])
default_sample_width = get_sample_size(paInt16)
current_energy_threshold = 300
dynamic_energy_adjustment_damping = 0.15
dynamic_energy_ratio = 1.5

wake_word = "jarvis"


def prep_mic(duration: float = 2.0) -> None:
    """
    Prepare the microphone for listening by adjusting it for ambient noise.

    :param duration: The duration to adjust the microphone for.
    :type duration: float
    :return: None
    """
    global current_energy_threshold
    chunk = 1024
    seconds_per_buffer = (chunk + 0.0) / default_sample_rate
    elapsed_time = 0

    p = PyAudio()
    stream = p.open(format=paInt16,
                    channels=1,
                    rate=default_sample_rate,
                    input=True,
                    frames_per_buffer=chunk)

    # adjust energy threshold until a phrase starts
    while True:
        elapsed_time += seconds_per_buffer
        if elapsed_time > duration:
            break
        buffer = stream.read(chunk)
        energy = audioop.rms(buffer, default_sample_width)  # energy of the audio signal

        # dynamically adjust the energy threshold using asymmetric weighted average
        damping = dynamic_energy_adjustment_damping ** seconds_per_buffer
        # account for different chunk sizes and rates
        target_energy = energy * dynamic_energy_ratio
        current_energy_threshold = current_energy_threshold * damping + target_energy * (1 - damping)
    p.terminate()
    logger.info("Microphone adjusted for ambient noise.")


def listen_to_user(silence_threshold=1) -> BytesIO:
    """
    Listen to the user and record their speech, stopping when there's silence.

    :return: The recorded audio data.
    :rtype: BytesIO
    """
    # Initialize PyAudio and create a stream
    p = PyAudio()
    stream = p.open(format=paInt16,
                    channels=1,
                    rate=default_sample_rate,
                    input=True,
                    frames_per_buffer=2048)

    audio_data = b''
    window_size = 10
    volume_buffer = []
    silence_duration = 0
    start = time.time()

    while True:
        # Read a chunk of audio data from the stream
        chunk = stream.read(2048)
        audio_data += chunk

        # Calculate the volume of the audio data in the chunk
        volume = frombuffer(chunk, dtype=int16)
        volume_buffer.append(average(abs(volume)))

        # Keep track of the average volume over the last window_size chunks
        if len(volume_buffer) > window_size:
            volume_buffer.pop(0)
        avg_volume = average(volume_buffer)

        # Keep track of the duration of silence
        if avg_volume < current_energy_threshold and time.time() - start > 3:
            silence_duration += len(chunk) / default_sample_rate
        else:
            silence_duration = 0

        # Stop recording if there is enough silence
        if silence_duration > silence_threshold:
            break

    # Close the stream and terminate PyAudio
    stream.stop_stream()
    stream.close()
    p.terminate()

    # Create an AudioData object from the recorded audio data

    # Fix this
    sample_rate = default_sample_rate
    sample_width = default_sample_width
    convert_rate = 16000

    if sample_rate != convert_rate:
        audio_data, _ = audioop.ratecv(audio_data, sample_width, 1, sample_rate, convert_rate, None)

    wave_file_data = BytesIO()
    # generate the WAV file contents
    wav_writer = wave.open(wave_file_data, "wb")
    try:  # note that we can't use context manager, since that was only added in Python 3.4
        wav_writer.setframerate(sample_rate)
        wav_writer.setsampwidth(sample_width)
        wav_writer.setnchannels(1)
        wav_writer.writeframes(audio_data)
        wave_file_data.seek(0)
    except Exception as e:
        # make sure resources are cleaned up
        logger.error(e)
        wav_writer.close()
    return wave_file_data


def convert_to_text(audio: BytesIO) -> str:
    """
    Convert the given audio data to text using speech recognition.

    :param audio: The audio data to convert.
    :type audio: BytesIO
    :return: str, the recognized text from the audio
    :rtype: str
    """
    dir_path = "whisper_models"
    if getattr(sys, 'frozen', False):
        dir_path = os.path.join(sys._MEIPASS, "whisper_models")
    model = whisper.load_model("base.en", download_root=dir_path)
    array_audio, sampling_rate = read(audio)
    array_audio = array_audio.astype(float32)
    result = model.transcribe(array_audio)
    return result["text"]


def audio_capture_process(audio_queue, speaking, current_recording_tag):
    """
    Captures audio from the microphone and puts it in the audio queue.
    :param audio_queue:
    :param speaking:
    :param current_recording_tag:
    :return: None
    """
    current_tag = 0
    while True:
        if not speaking.is_set():
            audio_data = listen_to_user()
            if not speaking.is_set():
                while not current_recording_tag.empty():
                    current_tag = current_recording_tag.get()
                audio_queue.put((audio_data, current_tag))


def audio_processing_process(audio_queue, text_queue, speaking, current_transcribing_tag):
    """
    Transcribes audio from the audio queue and puts it in the text queue.
    :param audio_queue:
    :param text_queue:
    :param speaking:
    :param current_transcribing_tag:
    :return:
    """
    current_tag = 0
    while True:
        while not current_transcribing_tag.empty():
            current_tag = current_transcribing_tag.get()
        if not speaking.is_set():
            audio_data, audio_tag = audio_queue.get()
            if not speaking.is_set() and audio_tag == current_tag:
                text = convert_to_text(audio_data)
                if not speaking.is_set():
                    while not current_transcribing_tag.empty():
                        current_tag = current_transcribing_tag.get()
                    text_queue.put((text, current_tag))


def process_assistant_response(query, beeps_stop_event, interrupt_event):
    """
    Processes the assistant's response to the user's query.
    :param query:
    :param beeps_stop_event:
    :param interrupt_event:
    :return: History of the conversation.
    :rtype: list
    """
    new_history = [{"content": query, "role": "user"}]
    content, reason, tool_calls = stream_audio_response(
        stream_response(new_history), stop_audio_event=beeps_stop_event, skip=interrupt_event)
    if not interrupt_event.is_set():
        if tool_calls is None:
            new_history.append({"content": content, "role": "assistant"})
            logger.info("Response Content:"+" "+str( content))
        else:
            while tool_calls is not None and not interrupt_event.is_set():
                tool_stop_event = play_audio_file("audio_files/searching.wav", loops=7, blocking=False)
                new_history += use_tools(tool_calls, content)
                tool_stop_event.set()
                content, reason, tool_calls = stream_audio_response(
                    stream_response(new_history, keep_last_history=True), stop_audio_event=beeps_stop_event)
            if tool_calls is None:
                new_history.append({"content": content, "role": "assistant"})
    if interrupt_event.is_set():
        logger.info("Interrupted")
        new_history.append({"content": "Interupted by user.", "role": "system"})
    return new_history


def converse(memory, interrupt_event=None, ready_event=None):
    """
    Converse with the user.
    :param memory:
    :type memory: int
    :param interrupt_event:
    :type interrupt_event: multiprocessing.Event
    :param ready_event:
    :type ready_event: multiprocessing.Event
    :return: None
    """
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    if interrupt_event is None:
        interrupt_event = multiprocessing.Event()
    if ready_event is None:
        ready_event = multiprocessing.Event()
        ready_event.set()

    audio_queue = multiprocessing.Queue()
    text_queue = multiprocessing.Queue()

    speaking = multiprocessing.Event()
    current_recording_tag = multiprocessing.Queue()
    current_transcribing_tag = multiprocessing.Queue()

    capture_process = multiprocessing.Process(target=audio_capture_process,
                                              args=(audio_queue, speaking, current_recording_tag))
    processing_process = multiprocessing.Process(target=audio_processing_process,
                                                 args=(audio_queue, text_queue, speaking, current_transcribing_tag))

    capture_process.start()
    processing_process.start()

    # Rolling buffer to store text
    transcript = []
    timestamps = []

    schedule_refresh_assistant()
    play_audio_file("audio_files/tone_one.wav", blocking=False)
    tone = time.time()
    while audio_queue.empty() and time.time() - tone < 10:
        pass
    ready_event.set()
    last_response_time = None
    current_tag = 1
    current_transcribing_tag.put(current_tag)
    current_recording_tag.put(current_tag)
    while True:
        interrupt_event.clear()
        text, assigned_tag = text_queue.get()
        if bool(re.search('[a-zA-Z0-9]', text)) and assigned_tag == current_tag:
            transcript.append(text)
            timestamps.append(time.time())

        # Clean up old entries
        current_time = time.time()
        while transcript and timestamps[0] < current_time - memory:
            transcript.pop(0)
            timestamps.pop(0)
        transcript_long = len("\n".join(transcript)) > 25
        if last_response_time is None:
            time_since_last_response = 100000000
        else:
            time_since_last_response = time.time() - last_response_time
        # Check for "jarvis" and print buffer if found
        if wake_word in text.lower() or (time_since_last_response < 15 and transcript_long):
            logger.info(" - - Checking if what has been said was directed at me...")
            logger.info("\n".join(transcript))
            directed_at_results = check_for_directed_at_me(transcript)
            directed_at_results = [x for x in directed_at_results if x <= 1.0]
            if len(directed_at_results) == 0:
                probably_at_me = False
            else:
                target_intended = 0.8
                probably_at_me = average(directed_at_results) > target_intended
            logger.info("Probability at me: " + str(directed_at_results))
            if probably_at_me:
                completion_results = check_for_completion(transcript)
                logger.info("Probability of completion:" + str(completion_results))
                completion_results = [x for x in completion_results if x <= 1.0]
                target_completion = 0.8
                if len(completion_results) == 0:
                    completed = False
                else:
                    completed = mean(completion_results) > target_completion
                if not completed:
                    additions = 0
                    max_time = 25
                    max_additions = 3
                    current_time = time.time()
                    while time.time() - current_time < max_time and additions < max_additions:
                        if not text_queue.empty():
                            text, assigned_tag = text_queue.get()
                            if bool(re.search('[a-zA-Z0-9]', text)) and assigned_tag == current_tag:
                                logger.info(" - - - Adding to transcript: " + text)
                                transcript.append(text)
                                timestamps.append(time.time())
                                additions += 1
                                completion_results = check_for_completion(transcript)
                                logger.info("Still checking probability of completion:"+" "+str( completion_results))
                                completion_results = [x for x in completion_results if x <= 1.0]
                                if len(completion_results) == 0:
                                    completed = False
                                else:
                                    completed = mean(completion_results) > target_completion
                                if completed:
                                    additions = max_additions
                speaking.set()
                beeps_stop_event = play_audio_file("audio_files/beeps.wav", loops=7, blocking=False)
                extracted_query = extract_query(transcript)
                logger.info("Query extracted: " + extracted_query)
                if not interrupt_event.is_set():
                    new_history = process_assistant_response(extracted_query, beeps_stop_event, interrupt_event)
                logger.info("Resolving...")
                resolve_response(new_history)
                logger.info("Resolved")
                if interrupt_event.is_set():
                    logger.info("Interrupted")
                    interrupt_event.clear()

                transcript = []
                timestamps = []
                current_tag += 1
                current_transcribing_tag.put(current_tag)
                current_recording_tag.put(current_tag)
                speaking.clear()
                while not audio_queue.empty():
                    audio_queue.get()
                while not text_queue.empty():
                    text_queue.get()
                last_response_time = time.time()
                play_audio_file("audio_files/tone_one.wav", blocking=False, delay=2)



