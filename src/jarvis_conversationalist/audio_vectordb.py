import os
import time
import fcntl
import warnings

import faiss
import numpy as np
import sqlite3


def normalize_embedding(embedding):
    """Normalize an embedding vector to unit length."""
    return embedding / np.linalg.norm(embedding)


class LocalAudioDB:

    def __init__(self, path, embedding_dim=256):
        if not os.path.exists(path):
            os.mkdir(path)
        db_path = os.path.join(path, 'embeddings_index.db')
        faiss_path = os.path.join(path, 'faiss_index.bin')
        lock_path = os.path.join(path, 'localaudiodb.lock')
        timestamp_path = os.path.join(path, 'localaudiodb_last_modified.txt')
        self.db_path = db_path
        self.faiss_path = faiss_path
        self.lock_path = lock_path
        self.timestamp_path = timestamp_path
        self.embedding_dim = embedding_dim
        self.conn = sqlite3.connect(db_path, check_same_thread=False)
        self.cursor = self.conn.cursor()
        self.initialize_db()
        self.index = faiss.IndexFlatL2(embedding_dim)
        self.load_or_reload_index()
        self.last_update_time = self.get_last_modified_time()
        self.lock_file = None

    def initialize_db(self):
        self.cursor.execute('''CREATE TABLE IF NOT EXISTS embedding_index 
                               (id TEXT PRIMARY KEY, faiss_index INTEGER, embedding BLOB, normal_embedding BLOB)''')
        self.conn.commit()
        self.cursor.execute('''CREATE TABLE IF NOT EXISTS used_ids
                                 (id TEXT PRIMARY KEY)''')
        self.conn.commit()

    def acquire_lock(self):
        self.lock_file = open(self.lock_path, 'w')
        fcntl.flock(self.lock_file, fcntl.LOCK_EX)

    def release_lock(self):
        fcntl.flock(self.lock_file, fcntl.LOCK_UN)
        self.lock_file.close()

    def update_last_modified_time(self):
        with open(self.timestamp_path, 'w') as f:
            f.write(str(time.time()))

    def get_last_modified_time(self):
        if not os.path.exists(self.timestamp_path):
            return None
        with open(self.timestamp_path, 'r') as f:
            return float(f.read())

    def check_for_updates(self):
        current_update_time = self.get_last_modified_time()
        if current_update_time != self.last_update_time:
            self.reload_index_from_db()
            self.last_update_time = current_update_time

    def get_largest_id(self):
        # check max id between used_ids and embedding_index
        self.cursor.execute("SELECT MAX(id) AS max_id FROM ("
                            " SELECT id FROM used_ids UNION ALL SELECT id FROM embedding_index);")
        result = self.cursor.fetchone()
        return result[0] if result else None

    def add_embedding(self, audio_id, embedding):
        self.acquire_lock()
        try:
            self.check_for_updates()
            index_id = self.index.ntotal
            normal_embedding = normalize_embedding(embedding)
            self.index.add(np.array([normal_embedding]))
            self.serialize_faiss_index()  # Save FAISS index after adding
            # Convert numpy array to blob
            embedding_blob = embedding.tobytes()
            normal_embedding_blob = normal_embedding.tobytes()
            self.cursor.execute("INSERT INTO embedding_index (id, faiss_index, embedding, normal_embedding) "
                                "VALUES (?, ?, ?, ?)",
                                (str(audio_id), index_id, embedding_blob, normal_embedding_blob))
            self.conn.commit()
            self.update_last_modified_time()
        finally:
            self.release_lock()

    def get_embedding(self, audio_id):
        self.acquire_lock()
        try:
            self.check_for_updates()
            self.cursor.execute("SELECT id, faiss_index, embedding FROM embedding_index WHERE id = ?", (str(audio_id),))
            id, faiss_index, embedding_blob = self.cursor.fetchone()
            embedding = np.frombuffer(embedding_blob, dtype=np.float32)
            return id, faiss_index, embedding
        finally:
            self.release_lock()

    def query_embeddings(self, audio_id_snippet):
        self.acquire_lock()
        try:
            self.check_for_updates()
            self.cursor.execute("SELECT id, faiss_index, embedding FROM embedding_index WHERE id "
                                "LIKE ?", ('%' + audio_id_snippet + '%',))
            results = self.cursor.fetchall()
            if results:
                for i, result in enumerate(results):
                    id, faiss_index, embedding_blob = result
                    embedding = np.frombuffer(embedding_blob, dtype=np.float32)
                    results[i] = (id, faiss_index, embedding)
                return results
        finally:
            self.release_lock()

    def remove_embedding(self, audio_id):
        self.acquire_lock()
        try:
            # Find the FAISS index of the audio ID
            self.cursor.execute("SELECT faiss_index FROM embedding_index WHERE id = ?", (str(audio_id),))
            result = self.cursor.fetchone()
            if result is None:
                return None

            self.cursor.execute("DELETE FROM embedding_index WHERE id = ?",
                                (str(audio_id),))
            self.conn.commit()
            self.cursor.execute("INSERT INTO used_ids (id) VALUES (?)", (str(audio_id),))
            self.conn.commit()

            self.rebuild_faiss_index()
            self.serialize_faiss_index()
            # Update the last modified time
            self.update_last_modified_time()
        finally:
            self.release_lock()
        return True

    def rebuild_faiss_index(self):
        # Get all embeddings from the SQLite database
        self.cursor.execute("SELECT normal_embedding FROM embedding_index ORDER BY faiss_index ASC")
        embeddings = self.cursor.fetchall()
        if embeddings:
            embeddings_array = np.array([np.frombuffer(e[0], dtype=np.float32) for e in embeddings])
            self.index = faiss.IndexFlatL2(self.embedding_dim)
            self.index.add(embeddings_array)

    def search_embeddings(self, query_embedding, k=5):
        query_embedding = normalize_embedding(query_embedding)
        distances, indices = self.index.search(np.array([query_embedding]), k)
        return distances, indices

    def find_closest_embedding(self, query_embedding):
        if self.index.ntotal == 0:
            return None, None, None
        query_embedding = normalize_embedding(query_embedding)

        distances, indices = self.index.search(np.array([query_embedding]), 1)

        closest_index = indices[0][0]
        closest_distance = distances[0][0]

        # Retrieve the corresponding ID and embedding from the SQLite database
        self.cursor.execute("SELECT id, embedding FROM embedding_index WHERE faiss_index = ?", (int(closest_index),))
        result = self.cursor.fetchone()
        if result:
            closest_id, embedding_blob = result
            closest_embedding = np.frombuffer(embedding_blob, dtype=np.float32)
            return closest_id, closest_embedding, closest_distance
        else:
            if self.index.ntotal == 1:
                return None, None, None
            raise Exception("FAISS index and SQLite database are out of sync.")

    def serialize_faiss_index(self):
        faiss.write_index(self.index, self.faiss_path)

    def load_or_reload_index(self):
        if os.path.exists(self.faiss_path):
            self.index = faiss.read_index(self.faiss_path)
            if self.index.ntotal-1 != self.get_sqlite_count() and self.index.ntotal >1:
                self.acquire_lock()
                try:
                    warnings.warn("Warning: FAISS index and SQLite database count mismatch. "
                                  "Reloading FAISS index from SQLite: " + str(self.index.ntotal-1) + " " +
                                  str(self.get_sqlite_count()))
                    self.reload_index_from_db()
                    self.serialize_faiss_index()
                finally:
                    self.release_lock()
        else:
            self.reload_index_from_db()

    def get_sqlite_count(self):
        self.cursor.execute("SELECT COUNT(*) FROM embedding_index")
        return self.cursor.fetchone()[0]

    def reload_index_from_db(self):
        self.cursor.execute("SELECT normal_embedding FROM embedding_index ORDER BY faiss_index ASC")
        embeddings = self.cursor.fetchall()
        if embeddings:
            embeddings_array = np.array([np.frombuffer(e[0], dtype=np.float32) for e in embeddings])
            self.index = faiss.IndexFlatL2(self.embedding_dim)
            self.index.add(embeddings_array)

    def close(self):
        self.conn.close()


