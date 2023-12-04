import wave
import time
from io import BytesIO
from numpy import frombuffer, int16, average, isnan, concatenate

from .logger_config import get_logger

logger = get_logger()


def prep_mic(duration: float = 2.0) -> int:
    """
    Prepare the microphone for listening by adjusting it for ambient noise.
    :param duration: The duration to adjust the microphone for.
    :return: Adjusted energy
    """
    import sounddevice as sd
    frames_per_buffer = 1024
    sample_rate = 16000
    seconds_per_buffer = frames_per_buffer / sample_rate
    elapsed_time = 0
    total_volume = 0
    count = 0

    with sd.InputStream(samplerate=sample_rate, channels=1, dtype='int16') as stream:
        stream.start()
        while elapsed_time < duration:
            chunk, overflowed = stream.read(frames_per_buffer)
            if overflowed:
                logger.warning("Input overflow detected")
            if not isnan(chunk).any():  # Check for NaN values
                chunk = chunk.astype(int16).tobytes()
                volume = average(abs(frombuffer(chunk, dtype=int16)))
                if not isnan(volume):
                    total_volume += volume
                    count += 1
            elapsed_time += seconds_per_buffer

    avgen = total_volume/count
    logger.info(f"Final Adjusted Microphone Energy: {avgen}")
    return int(avgen)


def listen_to_user(energy, silence_threshold=.5, maximum_seconds=120) -> BytesIO:
    """
    Listen to the user and record their speech, stopping when there's silence.

    :return: The recorded audio data.
    :rtype: BytesIO
    """
    import sounddevice as sd
    # Initialize PyAudio and create a stream
    sample_rate = 16000
    frames_per_buffer = 4096
    seconds_per_buffer = frames_per_buffer / sample_rate
    energy_factor = 1.2
    energy = energy*energy_factor
    with sd.InputStream(samplerate=sample_rate, channels=1, dtype='int16') as stream:

        audio_data = None
        window_size = round(silence_threshold * 2 / seconds_per_buffer)
        volume_buffer = []
        silence_duration = 0
        start = time.time()

        while time.time() - start < maximum_seconds:
            # Read a chunk of audio data from the stream

            chunk, overflowed = stream.read(frames_per_buffer)
            if overflowed:
                logger.warning("Input overflow detected")
            chunk = chunk.astype(int16)
            if audio_data is None:
                audio_data = chunk.flatten()
            else:
                audio_data = concatenate((audio_data, chunk.flatten()))

            # Calculate the volume of the audio data in the chunk
            volume_buffer.append(average(abs(chunk)))

            # Keep track of the average volume over the last window_size chunks
            if len(volume_buffer) > window_size:
                volume_buffer.pop(0)
            avg_volume = average(volume_buffer)
            # Keep track of the duration of silence
            if avg_volume < energy:
                silence_duration += seconds_per_buffer
            else:
                silence_duration = 0

            # Stop recording if there is enough silence
            if silence_duration > silence_threshold and time.time() - start > 2:
                break

        # Close the stream and terminate PyAudio

        if time.time() - start < 3:
            return None

        # Create an AudioData object from the recorded audio data
        audio_bytes_wave = BytesIO()
        with wave.open(audio_bytes_wave, 'wb') as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)  # 2 bytes per sample since dtype='int16'
            wf.setframerate(sample_rate)
            wf.writeframes(audio_data)

        return audio_bytes_wave


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