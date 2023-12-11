import os
import time
import threading
import soundfile as sf
from numpy import linspace, int16, sqrt, maximum, mean, square, frombuffer


def play_audio_file(file_path, blocking: bool = True, loops=1, delay: float = 0, destroy=False,
                    added_stop_event: threading.Event = None) -> threading.Event:
    """
    Play an audio file using pyaudio.

    :param file_path: path to the audio file or list of paths to play in sequence
    :type file_path: str or list of str
    :param blocking: whether the audio playback should block the main thread (default: True)
    :type blocking: bool
    :param loops: the number of times to loop the audio file (default: 1) or list of loop counts for each file
    :type loops: int or list of int
    :param delay: the delay in seconds before starting playback (default: 0)
    :type delay: float
    :param destroy: whether to destroy the file after playback (default: False) or list of destroy flags for each file
    :type destroy: bool or list of bool
    :param added_stop_event: an event to signal stopping the playback (only for non-blocking mode)
    :type added_stop_event: threading.Event
    :return: an event to signal stopping the playback (only for non-blocking mode)
    :rtype: threading.Event
    """
    stop_event = threading.Event()

    if blocking:
        time.sleep(delay)
        _play_audio_file_blocking(file_path, stop_event, loops, 0, destroy, added_stop_event)
    else:
        playback_thread = threading.Thread(target=_play_audio_file_blocking,
                                           args=(file_path, stop_event, loops, delay, destroy, added_stop_event))
        playback_thread.start()

    return stop_event


def _play_audio_file_blocking(file_path: str, stop_event: threading.Event, loops: int, delay: float, destroy: bool,
                              added_stop_event: threading.Event):
    """
    Play an audio file using pyaudio, blocking the calling thread until playback is complete or stopped.

    :param file_path: path to the audio file
    :type file_path: str
    :param stop_event: an event to signal stopping the playback
    :type stop_event: threading.Event
    :param loops: the number of times to loop the audio file
    :type loops: int
    :param delay: the delay in seconds before starting playback
    :type delay: float
    :param destroy: whether to destroy the file after playback
    :type destroy: bool
    :param added_stop_event: an event to signal stopping the playback
    :type added_stop_event: threading.Event
    """
    if isinstance(file_path, list):
        files_updated = []
        for file in file_path:
            files_updated.append(file)
        file_path = files_updated
        if isinstance(loops, list) and isinstance(destroy, list):
            for i, file in enumerate(file_path):
                _play_audio_file_blocking(file, stop_event, loops[i], delay, destroy[i], added_stop_event)
        elif isinstance(loops, list):
            for i, file in enumerate(file_path):
                _play_audio_file_blocking(file, stop_event, loops[i], delay, destroy, added_stop_event)
        else:
            for file in file_path:
                _play_audio_file_blocking(file, stop_event, loops, delay, destroy, added_stop_event)
        return
    else:
        chunk = 8192
        if isinstance(loops, list):
            loops = loops[0]  # Assumption: If loops is a list, take the first value

        # Wait for the specified delay
        if added_stop_event:
            added_stop_event.wait(timeout=delay)
        else:
            stop_event.wait(timeout=delay)
        import sounddevice as sd
        # Play the audio file
        for loop in range(loops):
            if not stop_event.is_set() or (added_stop_event and not added_stop_event.is_set()):
                data, fs = sf.read(file_path)
            sd.play(data, fs, latency=.25)
            while sd.get_stream().active:
                if stop_event.is_set() or (added_stop_event and added_stop_event.is_set()):
                    sd.stop()
                    break
                if added_stop_event:
                    added_stop_event.wait(timeout=.02)
                else:
                    stop_event.wait(timeout=.02)
            sd.stop()
        # Destroy the file if needed
        if destroy:
            os.remove(file_path)


def fade_out(data: bytes, fade_duration: int, rms_threshold: int = 1000) -> bytes:
    """
    Fade out the audio data.

    :param data: the audio data to fade out
    :type data: bytes
    :param fade_duration: the duration of the fade in samples
    :type fade_duration: int
    :param rms_threshold: the threshold for determining if audio is present
    :type rms_threshold: int
    :return: the audio data with a fade-out applied
    :rtype: bytes
    """

    def rms(aud_data):
        return sqrt(maximum(mean(square(aud_data)), 0))

    num_samples = len(data) // 2  # Divide by 2 for 16-bit audio samples
    fade_samples = min(num_samples, fade_duration)
    audio_data = frombuffer(data, dtype=int16).copy()  # Create a writeable copy of the array

    start_fade = 0
    for i in range(0, num_samples - fade_samples, fade_samples):
        if rms(audio_data[i:i + fade_samples]) < rms_threshold:
            start_fade = i
            break

    fade = linspace(1, 0, num_samples - start_fade).astype(int16)  # Convert fade array to int16
    audio_data[start_fade:] *= fade
    return audio_data.tobytes()
