import os
from io import BytesIO
from soundfile import read
from numpy import float32
import whisper
import warnings

from .logger_config import get_log_folder_path

warnings.filterwarnings("ignore", message="FP16 is not supported on CPU; using FP32 instead")

dir_root = get_log_folder_path()

dir_path = os.path.join(dir_root, "whisper_models")
if not os.path.exists(dir_path):
    os.mkdir(dir_path)
model = whisper.load_model("base.en", download_root=dir_path)


def convert_to_text(audio: BytesIO) -> str:
    """
    Convert the given audio data to text using speech recognition.

    :param audio: The audio data to convert.
    :type audio: BytesIO
    :return: str, the recognized text from the audio
    :rtype: str
    """
    array_audio, sampling_rate = read(audio)
    array_audio = array_audio.astype(float32)
    result = model.transcribe(array_audio)
    return result["text"]


def audio_processing_process(audio_queue, text_queue, speaking):
    """
    Transcribes audio from the audio queue and puts it in the text queue.
    :param audio_queue:
    :param text_queue:
    :param speaking:
    :return:
    """
    try:
        while True:
            if not speaking.is_set():
                audio_data = audio_queue.get()
                if not speaking.is_set():
                    text = convert_to_text(audio_data)
                    if not speaking.is_set():
                        text_queue.put(text)
    except KeyboardInterrupt:
        return
