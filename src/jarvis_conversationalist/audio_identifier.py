import os
import uuid
import warnings
from io import BytesIO

warnings.filterwarnings("ignore", message=".*audio._backend.set_audio_backend.*")

from pyannote.audio import Pipeline
from .audio_vectordb import LocalAudioDB


def db_speaker_id(speaker_id):
    """
    This method converts a speaker ID to the format used in the speaker databases.
    """
    speaker_id = "Unknown Speaker " + str(int(speaker_id.replace("Unknown Speaker ", ""))).zfill(9)
    return speaker_id


def pretty_speaker_id(speaker_id):
    """
    This method converts a speaker ID to a more readable format.
    """
    speaker_id = "Unknown Speaker " + str(int(speaker_id.replace("Unknown Speaker ", "")))
    return speaker_id


def pretty_known_speaker_id(speaker_id):
    """
    This method converts a speaker ID to a more readable format.
    """
    speaker_id = speaker_id.split(" -- ")[0]
    return speaker_id


class SpeakerIdentifier:
    """
    This class is responsible for identifying speakers in an audio stream. It uses a pretrained model from Pyannote
     for speaker diarization and stores known and unknown speaker embeddings in separate databases.
    """

    def __init__(
        self,
        persist_directory: str = "speaker_database",
        similar_speaker_threshold: float = 1,
    ):
        """
        The constructor for the SpeakerIdentifier class. It initializes the class with a directory to persist speaker
        data and a threshold for determining if a speaker is similar to a known speaker.
        """
        self.similar_speaker_threshold = similar_speaker_threshold
        self.known_speakers_path = os.path.join(persist_directory, "known_speakers_db")
        self.known_speakers = LocalAudioDB(self.known_speakers_path)
        self.unknown_speakers_path = os.path.join(persist_directory, "unknown_speakers_db")
        self.unknown_speakers = LocalAudioDB(self.unknown_speakers_path)
        self.pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization-3.1",
                                                 use_auth_token="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"
                                                 )

    def get_next_unknown_speaker_id(self):
        """
        This method retrieves the next available ID for an unknown speaker from the unknown speakers database.

        :return: the speaker ID
        :rtype: str
        """
        largest_id = self.unknown_speakers.get_largest_id()
        if largest_id is None:
            largest_id = "Unknown Speaker 0"
        largest_id = "Unknown Speaker "+str(int(largest_id.replace("Unknown Speaker ", ""))+1)
        return largest_id

    def get_last_unknown_speaker_id(self):
        """
        This method retrieves the last added ID for an unknown speaker from the unknown speakers database.

        :return: the speaker ID
        :rtype: str
        """
        largest_id = self.unknown_speakers.get_largest_id()
        if largest_id is None:
            largest_id = "Unknown Speaker 0"
        largest_id = "Unknown Speaker "+str(int(largest_id.replace("Unknown Speaker ", "")))
        return largest_id

    def check_known_speakers(self, embedding):
        """
        This method checks if a given speaker embedding is similar to any known speaker embeddings.
        If a similar speaker is found, their ID is returned.

        :param embedding: the speaker embedding to check
        :type embedding: numpy array
        :return: the speaker ID
        :rtype: str
        """
        closest_id, closest_embedding, closest_distance = self.known_speakers.find_closest_embedding(embedding)
        if closest_distance:
            if int(closest_distance) < self.similar_speaker_threshold:
                closest_id = pretty_known_speaker_id(closest_id)
                return closest_id
        return None

    def add_known_speaker(self, embedding, name, new=False):
        """
        This method adds a known speaker to the known speakers database.

        :param embedding: the speaker embedding
        :type embedding: numpy array
        :param name: the speaker name
        :type name: str
        :param new: whether this is a new speaker
        :type new: bool
        """
        name = name.strip()+" -- "+str(uuid.uuid4())
        if new:
            result = self.known_speakers.query_embeddings(name)
            if result:
                raise Exception("Speaker name already exists.")
        self.known_speakers.add_embedding(name, embedding)

    def remove_known_speaker(self, name):
        """
        This method removes a known speaker from the known speakers' database.

        :param name: the speaker name
        :type name: str
        """
        results = self.known_speakers.query_embeddings(name + " -- ")
        if results:
            for result in results:
                self.known_speakers.remove_embedding(result[0])
        else:
            raise Exception("Speaker name does not exist.")

    def get_add_unknown_speaker(self, embedding):
        """
        This method checks if a given speaker embedding is similar to any unknown speaker embeddings.
        If a similar speaker is found, their ID is returned. If not, a new speaker ID is generated and the embedding
        is added to the unknown speakers' database.

        :param embedding: the speaker embedding to check
        :type embedding: numpy array
        :return: the speaker ID
        :rtype: str
        """
        # cut the audio data to the start and end times
        # from pydub import AudioSegment
        # audio_data_io.seek(0)
        # audio_data = AudioSegment.from_file(audio_data_io, format="wav")
        # audio_data = audio_data[start*1000:end*1000]
        # audio_data_io_clip = BytesIO()
        # audio_data.export(audio_data_io_clip, format="wav")
        # audio_data_io_clip.seek(0)
        # diarization, embeddings = self.pipeline(audio_data_io_clip, return_embeddings=True)
        # embedding = embeddings[0]
        closest_id, closest_embedding, closest_distance = self.unknown_speakers.find_closest_embedding(embedding)
        if closest_distance is not None:
            if int(closest_distance) < self.similar_speaker_threshold:
                return pretty_speaker_id(closest_id)
        speaker_id = self.get_next_unknown_speaker_id()
        self.unknown_speakers.add_embedding(db_speaker_id(speaker_id), embedding)
        return pretty_speaker_id(speaker_id)

    def get_unknown_embedding(self, speaker_id):
        """
        This method retrieves the embedding of an unknown speaker from the unknown speakers' database.

        :param speaker_id: the speaker ID
        :type speaker_id: str
        :return: the speaker embedding
        :rtype: numpy array
        """
        id, faiss_index, embedding = self.unknown_speakers.get_embedding(db_speaker_id(speaker_id))
        return embedding

    def remove_unknown_speaker(self, speaker_id):
        """
        This method removes an unknown speaker from the unknown speakers database.

        :param speaker_id: the speaker ID
        :type speaker_id: str
        """
        self.unknown_speakers.remove_embedding(db_speaker_id(speaker_id))

    def get_speakers(self, audio_data_io):
        """
        This method takes an audio data stream as input and returns a timeline of speakers.
        It uses the Pyannote pipeline to perform speaker diarization and retrieve speaker embeddings.
        These embeddings are then checked against the known and unknown speaker databases to identify the speakers.

        :param audio_data_io: the audio data stream to process
        :type audio_data_io: BytesIO
        :return: the timeline of speakers
        :rtype: list
        """
        audio_data_io.seek(0)

        # Process the NumPy array with Pyannote
        diarization, embeddings = self.pipeline(audio_data_io, return_embeddings=True)

        # combine embeddings with diarization.labels() into dict
        speakers = {}
        for i, speaker in enumerate(diarization.labels()):
            speakers[speaker] = embeddings[i]

        # Create a timeline
        timeline = []
        found_speakers = {}
        for turn, _, speaker_code in diarization.itertracks(yield_label=True):
            start, end = turn.start, turn.end
            if speaker_code in found_speakers:
                continue
            else:
                speaker_id = self.check_known_speakers(speakers[speaker_code])
                if speaker_id is None:
                    speaker_id = self.get_add_unknown_speaker(speakers[speaker_code])
                found_speakers[speaker_code] = speaker_id
            timeline.append((start, end, speaker_id))
        return timeline

# from logger_config import get_log_folder_path
# dir_root = get_log_folder_path()
# self = SpeakerIdentifier(persist_directory=dir_root)
# while self.get_next_unknown_speaker_id() != "Unknown Speaker 1":
#     self.remove_unknown_speaker(self.get_last_unknown_speaker_id())
# # read test.wav into audio_data_io
# audio_data_io = BytesIO()
# audio_data_io.write(open("1.wav", "rb").read())
# audio_data_io.seek(0)
# timeline = self.get_speakers(audio_data_io)
# print(timeline)
# audio_data_io = BytesIO()
# audio_data_io.write(open("2.wav", "rb").read())
# audio_data_io.seek(0)
# timeline = self.get_speakers(audio_data_io)
# print(timeline)
# audio_data_io = BytesIO()
# audio_data_io.write(open("3.wav", "rb").read())
# audio_data_io.seek(0)
# timeline = self.get_speakers(audio_data_io)
# print(timeline)
# audio_data_io = BytesIO()
# audio_data_io.write(open("4.wav", "rb").read())
# audio_data_io.seek(0)
# timeline = self.get_speakers(audio_data_io)
# print(timeline)
