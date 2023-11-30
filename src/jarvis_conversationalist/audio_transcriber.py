import os
import sys
import importlib.resources as pkg_resources
cuda_capable = False
if sys.platform == "linux":
    try:
        cublas_path = str(pkg_resources.files('nvidia').joinpath('cublas', 'lib'))
        cudnn_path = str(pkg_resources.files('nvidia').joinpath('cudnn', 'lib'))
        if 'LD_LIBRARY_PATH' not in os.environ:
            print(f"\nRun this to run Jarvis:\n"
                  f"export LD_LIBRARY_PATH={cublas_path}:{cudnn_path}\n")
        else:
            if os.environ['LD_LIBRARY_PATH'].find(f"{cublas_path}:{cudnn_path}") < 0:
                print(f"\nLD_LIBRARY_PATH needs addition of {cublas_path}:{cudnn_path}\n")
            else:
                cuda_capable = True
    finally:
        from torch.cuda import is_available
        if cuda_capable and is_available():
            from faster_whisper import WhisperModel, download_model
            model_size = "large-v2"
            # download_model("large-v2", dir_path)
            model = WhisperModel(model_size, device="cuda", compute_type="float16")
            import logging

            logging.basicConfig()
            logging.getLogger("faster_whisper").setLevel(logging.DEBUG)
        else:
            import whisper
else:
    import whisper

import time
from io import BytesIO
from soundfile import read
from numpy import float16
import warnings

from .audio_identifier import SpeakerIdentifier
from .logger_config import get_log_folder_path


warnings.filterwarnings("ignore", message="FP16 is not supported on CPU; using FP32 instead")

dir_root = get_log_folder_path()

dir_path = os.path.join(dir_root, "whisper_models")
if not os.path.exists(dir_path):
    os.mkdir(dir_path)

speaker_pipeline = SpeakerIdentifier(persist_directory=dir_root)


def convert_to_text(audio: BytesIO, text_only=False) -> str or list:
    """
    Convert the given audio data to text using speech recognition.

    :param audio: The audio data to convert.
    :type audio: BytesIO
    :param text_only: Whether to return only the text or the full result.
    :type text_only: bool
    :return: str, the recognized text from the audio
    :rtype: str or list
    """
    audio.seek(0)
    if cuda_capable:
        result = {"text": "", "segments": []}
        segments, info = model.transcribe(audio, beam_size=5, language="en")

        for segment in segments:
            print(segment)
            new_seg = {'start': segment.start, 'end': segment.end, 'text': segment.text}
            result['segments'].append(new_seg)
            result['text'] += (" " + segment.text)
    else:
        array_audio, sampling_rate = read(audio)
        array_audio = array_audio.astype(float16)
        result = model.transcribe(array_audio)

    if text_only:
        return result["text"]
    else:
        return result


def fuse_text_and_speakers(whisper_output, speakers):
    segments = whisper_output['segments']
    annotated_text = ""
    last_speaker = None
    segment_index = 0

    # Function to append text with the speaker label
    def append_with_speaker(text, speaker):
        nonlocal annotated_text, last_speaker
        if speaker != last_speaker:
            annotated_text += f"\n[{speaker}]: " if text.strip() else ""
            last_speaker = speaker
        annotated_text += text + " "

    # Iterate through each segment and assign speakers
    while segment_index < len(segments):
        segment = segments[segment_index]
        overlapping_speakers = [speaker_label for start, end, speaker_label in speakers if
                                start <= segment['end'] and end >= segment['start']]

        if not overlapping_speakers:
            append_with_speaker(segment['text'], 'Unknown Speaker')
        elif len(overlapping_speakers) == 1:
            append_with_speaker(segment['text'], overlapping_speakers[0])
        else:
            append_with_speaker(segment['text'], 'Multiple Speakers')

        segment_index += 1

    return annotated_text.strip()


def audio_processing_thread(audio_queue, text_queue, speaking, stop_event):
    """
    Transcribes audio from the audio queue and puts it in the text queue.
    :param audio_queue:
    :param text_queue:
    :param speaking:
    :param stop_event:
    :return:
    """
    try:
        while stop_event.is_set() is False:
            audio_data, ts = audio_queue.get()
            if not speaking.is_set() and audio_data is not None:
                print(audio_data is not None)
                text = convert_to_text(audio_data)
                timestamp = time.time()
                if not speaking.is_set() and audio_data is not None:
                    #speakers = speaker_pipeline.get_speakers(audio_data)
                    #text = fuse_text_and_speakers(text, speakers)
                    if not speaking.is_set():
                        text_queue.put((text, ts))
            if audio_data is None:
                text_queue.put(("", ts))
    except KeyboardInterrupt:
        return


# # Example usage
# whisper_output = {
#     "segments": [
#         {"start": 0.0, "end": 3.0, "text": "Hello, how are you?"},
#         {"start": 3.0, "end": 6.0, "text": "I'm fine, thank you."}
#     ]
# }
# whisper_output = {
#     "segments": [
#         {"start": 0.0, "end": 2.0, "text": "Good morning, everyone."},
#         {"start": 2.0, "end": 4.0, "text": "Today, we are discussing our new project."},
#         {"start": 4.0, "end": 6.0, "text": "I think it's a great opportunity to grow."},
#         {"start": 6.0, "end": 8.0, "text": "Absolutely, and with the right strategy, we can succeed."},
#         {"start": 8.0, "end": 10.0, "text": "Let's not forget about the challenges ahead."},
#         {"start": 10.0, "end": 12.0, "text": "Yes, especially the tight deadlines."},
#         {"start": 12.0, "end": 12.5, "text": "We'll need to collaborate effectively."},
#         {"start": 14.0, "end": 16.0, "text": "I agree, teamwork is key here."}
#     ]
# }
#
# speakers = [
#     (0.0, 3.0, 'SPEAKER_01'),  # Overlaps with SPEAKER_02
#     (2.0, 5.0, 'SPEAKER_02'),  # Overlaps with SPEAKER_01 and SPEAKER_03
#     (4.0, 7.0, 'SPEAKER_03'),  # Overlaps with SPEAKER_02
#     (7.0, 9.0, 'SPEAKER_04'),
#     (9.0, 11.0, 'SPEAKER_05'),
#     # Omitting explicit 'Unknown Speaker' label
#     (13.0, 15.0, 'SPEAKER_06')
#     # No speaker data for segments from 11.0 to 13.0 and the last segment
# ]
#
# annotated_text = fuse_text_and_speakers(whisper_output, speakers)
# print(annotated_text)
