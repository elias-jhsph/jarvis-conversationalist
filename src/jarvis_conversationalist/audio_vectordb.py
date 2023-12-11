import os
import time
import fcntl
import atexit

import faiss
import numpy as np
import sqlite3

from faiss import logger

logger.setLevel("CRITICAL")


def normalize_embedding(embedding):
    """Normalize an embedding vector to unit length."""
    return embedding / np.linalg.norm(embedding)


class LocalAudioDB:

    def __init__(self, path, embedding_dim=256):
        if not os.path.exists(path):
            os.mkdir(path)
        db_path = os.path.join(path, 'embeddings_index.db')
        lock_path = os.path.join(path, 'localaudiodb.lock')
        self.db_path = db_path
        self.lock_path = lock_path
        self.embedding_dim = embedding_dim
        self.conn = sqlite3.connect(db_path, check_same_thread=False)
        self.cursor = self.conn.cursor()
        self.initialize_db()
        self.index = faiss.IndexFlatL2(embedding_dim)
        self.last_update_time = self.get_last_modified_time()
        if self.last_update_time is None:
            self.acquire_lock(check_for_updates=False)
            try:
                self.update_last_modified_time()
            finally:
                self.release_lock()
        self.lock_file = None
        self.reload_index_from_db()
        atexit.register(self.close)
        atexit.register(self.try_release_lock)

    def initialize_db(self):
        self.cursor.execute('''CREATE TABLE IF NOT EXISTS embedding_index 
                               (id TEXT PRIMARY KEY, embedding BLOB, normal_embedding BLOB)''')
        self.conn.commit()
        self.cursor.execute('''CREATE TABLE IF NOT EXISTS used_ids
                                 (id TEXT PRIMARY KEY)''')
        self.conn.commit()
        self.cursor.execute('''CREATE TABLE IF NOT EXISTS last_update_time
                                    (timestamp REAL)''')
        self.conn.commit()

    def acquire_lock(self, check_for_updates=True):
        self.lock_file = open(self.lock_path, 'w')
        fcntl.flock(self.lock_file, fcntl.LOCK_EX)
        if check_for_updates:
            self.check_for_updates()

    def release_lock(self):
        fcntl.flock(self.lock_file, fcntl.LOCK_UN)
        self.lock_file.close()

    def try_release_lock(self):
        try:
            self.release_lock()
        except:
            pass

    def update_last_modified_time(self):
        # Must have lock
        self.cursor.execute("DELETE FROM last_update_time")
        self.conn.commit()
        self.cursor.execute("INSERT INTO last_update_time (timestamp) VALUES (?)", (time.time(),))
        self.conn.commit()
        self.last_update_time = time.time()

    def get_last_modified_time(self):
        self.cursor.execute("SELECT timestamp FROM last_update_time")
        result = self.cursor.fetchone()
        return result[0] if result else None

    def check_for_updates(self):
        current_update_time = self.get_last_modified_time()
        if current_update_time > self.last_update_time:
            self.reload_index_from_db()
            self.last_update_time = current_update_time
        else:
            self.cursor.execute("SELECT COUNT(*) FROM embedding_index")
            result = self.cursor.fetchone()
            max_index = result[0] if result else None
            if max_index is not None and max_index != self.index.ntotal:
                self.try_release_lock()
                raise Exception("FAISS and SQLite database are out of sync: " +
                                str(max_index) + " " + str(self.index.ntotal))

    def get_largest_id(self):
        # check max id between used_ids and embedding_index
        self.cursor.execute("SELECT MAX(id) AS max_id FROM ("
                            " SELECT id FROM used_ids UNION ALL SELECT id FROM embedding_index);")
        result = self.cursor.fetchone()
        return result[0] if result else None

    def add_embedding(self, audio_id, embedding):
        self.acquire_lock()
        try:
            normal_embedding = normalize_embedding(embedding)
            embedding_blob = embedding.tobytes()
            normal_embedding_blob = normal_embedding.tobytes()
            self.cursor.execute("INSERT INTO embedding_index (id, embedding, normal_embedding) "
                                "VALUES (?, ?, ?)",
                                (str(audio_id), embedding_blob, normal_embedding_blob))
            self.conn.commit()
            self.update_last_modified_time()
            self.reload_index_from_db()
        finally:
            self.release_lock()

    def get_embedding(self, audio_id):
        self.check_for_updates()
        self.cursor.execute("SELECT id, embedding FROM embedding_index WHERE id = ?", (str(audio_id),))
        id, embedding_blob = self.cursor.fetchone()
        embedding = np.frombuffer(embedding_blob, dtype=np.float32)
        return id, embedding

    def query_embeddings(self, audio_id_snippet):
        self.check_for_updates()
        self.cursor.execute("SELECT id, embedding FROM embedding_index WHERE id "
                            "LIKE ?", ('%' + audio_id_snippet + '%',))
        results = self.cursor.fetchall()
        if results:
            for i, result in enumerate(results):
                id, embedding_blob = result
                embedding = np.frombuffer(embedding_blob, dtype=np.float32)
                results[i] = (id, embedding)
            return results

    def remove_embedding(self, audio_id):
        self.acquire_lock()
        try:
            self.cursor.execute("DELETE FROM embedding_index WHERE id = ?",
                                (str(audio_id),))
            self.conn.commit()
            self.cursor.execute("INSERT INTO used_ids (id) VALUES (?)", (str(audio_id),))
            self.conn.commit()

            self.update_last_modified_time()
            self.reload_index_from_db()
        finally:
            self.release_lock()
        return True

    def search_embeddings(self, query_embedding, k=5):
        self.check_for_updates()
        query_embedding = normalize_embedding(query_embedding)
        distances, indices = self.index.search(np.array([query_embedding]), k)
        return distances, indices

    def find_closest_embedding(self, query_embedding, retry=3):
        self.acquire_lock()
        try:
            if self.index.ntotal == 0:
                return None, None, None
            normal_query_embedding = normalize_embedding(query_embedding)

            distances, indices = self.index.search(np.array([normal_query_embedding]), 1)

            closest_index = indices[0][0]
            closest_distance = distances[0][0]

            # Retrieve the corresponding ID by sorting the IDs in ascending order and selecting the ID at the index
            self.cursor.execute("SELECT id, normal_embedding FROM embedding_index ORDER BY id ASC LIMIT 1 OFFSET ?-1;",
                                (closest_index+1,))
            result = self.cursor.fetchone()
            if result:
                closest_id, embedding_blob = result
                closest_embedding = np.frombuffer(embedding_blob, dtype=np.float32)
                return closest_id, closest_embedding, closest_distance
            else:
                if self.index.ntotal == 1:
                    return None, None, None
                if retry > 0:
                    self.release_lock()
                    self.find_closest_embedding(query_embedding, retry=retry-1)
                else:
                    self.try_release_lock()
                    raise Exception("FAISS index and SQLite database are out of sync.")
        finally:
            self.release_lock()

    def reload_index_from_db(self):
        self.cursor.execute("SELECT normal_embedding FROM embedding_index ORDER BY id ASC")
        embeddings = self.cursor.fetchall()
        if embeddings:
            embeddings_array = np.array([np.frombuffer(e[0], dtype=np.float32) for e in embeddings])
            self.index = faiss.IndexFlatL2(self.embedding_dim)
            self.index.add(embeddings_array)

    def close(self):
        self.conn.close()


