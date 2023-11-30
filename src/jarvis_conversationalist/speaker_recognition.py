import os
from .logger_config import get_log_folder_path, get_logger
logger = get_logger()


dir_root = get_log_folder_path()

dir_path = os.path.join(dir_root, "whisper_models")
if not os.path.exists(dir_path):
    os.mkdir(dir_path)


def recognition_process(audio_queue, text_queue, speaking, stop_event):
    from .audio_identifier import SpeakerIdentifier
    speaker_pipeline = SpeakerIdentifier(persist_directory=dir_root)
    speaker_pipeline.speedup_if_able()
    while stop_event.is_set() is False:
        audio_data = audio_queue.get()
        if not speaking.is_set() and audio_data is not None:
            if not speaking.is_set():
                speakers = speaker_pipeline.get_speakers(audio_data)
                text_queue.put(speakers)
