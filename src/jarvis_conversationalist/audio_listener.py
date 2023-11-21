import wave
import time
import audioop
import atexit
from io import BytesIO
from numpy import frombuffer, int16, average
from pyaudio import PyAudio, paInt16, get_sample_size

from .logger_config import get_logger
logger = get_logger()

# Initialize the speech recognition module and microphone
paudio = PyAudio()
device_info = paudio.get_default_input_device_info()
paudio.terminate()
default_sample_rate = 16000
default_sample_width = get_sample_size(paInt16)
dynamic_energy_adjustment_damping = 0.15
dynamic_energy_ratio = 1.5


def prep_mic(energy=300, duration: float = 1.0) -> int:
    """
    Prepare the microphone for listening by adjusting it for ambient noise.

    :param energy: The energy threshold to adjust the microphone for.
    :type energy: int
    :param duration: The duration to adjust the microphone for.
    :type duration: float
    :return: energy
    :rtype: int
    """
    chunk = 1024
    seconds_per_buffer = (chunk + 0.0) / default_sample_rate
    elapsed_time = 0

    p = PyAudio()
    stream = p.open(format=paInt16,
                    channels=1,
                    rate=default_sample_rate,
                    input=True,
                    frames_per_buffer=chunk)
    atexit.register(stream.stop_stream)
    atexit.register(stream.close)
    atexit.register(p.terminate)

    # adjust energy threshold until a phrase starts
    while True:
        elapsed_time += seconds_per_buffer
        if elapsed_time > duration:
            break
        buffer = stream.read(chunk, exception_on_overflow=False)
        energy = audioop.rms(buffer, default_sample_width)  # energy of the audio signal

        # dynamically adjust the energy threshold using asymmetric weighted average
        damping = dynamic_energy_adjustment_damping ** seconds_per_buffer
        # account for different chunk sizes and rates
        target_energy = energy * dynamic_energy_ratio
        energy = energy * damping + target_energy * (1 - damping)
    stream.stop_stream()
    stream.close()
    p.terminate()
    logger.info("Microphone adjusted for ambient noise.")
    return energy


def listen_to_user(energy, silence_threshold=.5, maximum_seconds=120) -> BytesIO:
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
                    frames_per_buffer=4096)
    atexit.register(stream.stop_stream)
    atexit.register(stream.close)
    atexit.register(p.terminate)

    audio_data = b''
    window_size = 10
    volume_buffer = []
    silence_duration = 0
    start = time.time()

    while time.time() - start < maximum_seconds:
        # Read a chunk of audio data from the stream

        chunk = stream.read(4096, exception_on_overflow=False)
        audio_data += chunk

        # Calculate the volume of the audio data in the chunk
        volume = frombuffer(chunk, dtype=int16)
        volume_buffer.append(average(abs(volume)))

        # Keep track of the average volume over the last window_size chunks
        if len(volume_buffer) > window_size:
            volume_buffer.pop(0)
        avg_volume = average(volume_buffer)

        # Keep track of the duration of silence
        if avg_volume < energy:
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
    if time.time() - start < 3:
        return None

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


def audio_capture_process(audio_queue, speaking, stop_event):
    """
    Captures audio from the microphone and puts it in the audio queue.
    :param audio_queue:
    :param speaking:
    :param stop_event:
    :return: None
    """
    try:
        first = True
        level = prep_mic()
        while stop_event.is_set() is False:
            if not speaking.is_set():
                audio_data = listen_to_user(level)
                if not speaking.is_set() and not stop_event.is_set() and audio_data is not None:
                    audio_queue.put((audio_data, time.time()))
                if first:
                    first = False
                    audio_queue.put((None, time.time()))
    except KeyboardInterrupt:
        return