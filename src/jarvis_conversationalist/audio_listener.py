import wave
import time
from io import BytesIO
from numpy import frombuffer, int16, average, isnan

from .logger_config import get_logger

logger = get_logger()


def prep_mic(duration: float = 1.0) -> int:
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

    with sd.InputStream(samplerate=sample_rate, channels=2, dtype='int16') as stream:
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
    with sd.InputStream(samplerate=sample_rate, channels=1, dtype='int16') as stream:

        audio_data = b''
        window_size = round(silence_threshold * 2 / seconds_per_buffer)
        volume_buffer = []
        silence_duration = 0
        start = time.time()

        while time.time() - start < maximum_seconds:
            # Read a chunk of audio data from the stream

            chunk, overflowed = stream.read(frames_per_buffer)
            if overflowed:
                logger.warning("Input overflow detected")
            chunk = chunk.astype(int16).tobytes()
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
        wave_file_data = BytesIO()
        # generate the WAV file contents
        wav_writer = wave.open(wave_file_data, "wb")
        try:  # note that we can't use context manager, since that was only added in Python 3.4
            wav_writer.setframerate(sample_rate)
            wav_writer.setsampwidth(2)
            wav_writer.setnchannels(2)
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
                    print("Adding audio:", time.time())
                    audio_queue.put((audio_data, time.time()))
                if first:
                    first = False
                    audio_queue.put((None, time.time()))
    except KeyboardInterrupt:
        return