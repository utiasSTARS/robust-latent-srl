import threading
from queue import Queue

import numpy as np
import os
import sqlite3

from data_collection.utils import adapt_ndarray, convert_ndarray

class Storage(threading.Thread):
    """ This is a storage that uses SQLite3.
    Writing is on a separate thread to reduce blocking time while interacting with the environment.
    You should wait until all writes are flushed before closing the connection.

    NOTE: This only support one instance for a database.
    NOTE: Episode == Trajectory

    The database consists of 3 tables:
    - sample: (Sample ID, State, Action, Reward, Done)
    - sample_relation: (Sample ID, Episode, Timestep)
    - last_obs: (Episode, State)
    """
    def __init__(self, dbname):
        super(Storage, self).__init__()
        self._stop_event = threading.Event()

        sqlite3.register_adapter(np.ndarray, adapt_ndarray)
        sqlite3.register_converter("NDARRAY", convert_ndarray)
        self._dbname = dbname
        self._is_new = not os.path.exists(dbname)
        
        self._main_conn = sqlite3.connect(
            dbname, detect_types=sqlite3.PARSE_DECLTYPES)
        self._main_cursor = self._main_conn.cursor()

        self._counter = 0
        self._episode_counter = 0

        if self._is_new:
            print("{} not found... Creating new database".format(dbname))

            self._main_cursor.executescript("""
                CREATE TABLE sample(
                    sample_id INTEGER PRIMARY KEY,
                    curr_obs NDARRAY,
                    action NDARRAY,
                    reward REAL,
                    done bit
                );
                CREATE TABLE sample_relation(
                    episode INTEGER,
                    timestep INTEGER,
                    sample_id INTEGER,
                    PRIMARY KEY (episode, timestep),
                    FOREIGN KEY(sample_id) REFERENCES sample(sample_id)
                );
                CREATE TABLE last_obs(
                    episode INTEGER PRIMARY KEY,
                    last_obs NDARRAY
                );
                """)
            self._main_conn.commit()
        else:
            self._counter = self.get_num_transitions()
            self._episode_counter = self.get_num_episodes()

        self.reqs = Queue()

    def run(self):
        write_conn = sqlite3.connect(
            self._dbname, detect_types=sqlite3.PARSE_DECLTYPES)
        write_cursor = write_conn.cursor()
        while True:
            if self._stop_event.isSet() and self.reqs.empty():
                break
            request, args = self.reqs.get()
            try:
                write_cursor.execute(request, args)
                write_conn.commit()
            except sqlite3.IntegrityError:
                print("Conflicting entry")
        print("Closing connection")
        write_conn.close()

    def save_transition(self, episode, timestep, curr_obs, action, reward, done, next_obs):
        self.reqs.put(("insert into sample(curr_obs, action, reward, done, sample_id) values (?, ?, ?, ?, ?)", (curr_obs, action, reward, done, self._counter)))
        if done:
            self.reqs.put(("insert into last_obs(episode, last_obs) values (?, ?)", (self._episode_counter + episode, next_obs)))
        self.reqs.put(("insert into sample_relation(episode, timestep, sample_id) values (?, ?, ?)", (self._episode_counter + episode, timestep, self._counter)))
        self._counter += 1

    def get_transition(self, episode, timestep):
        self._main_cursor.execute("""
            SELECT curr_obs, action, reward, done
            FROM (
                SELECT sample_id
                FROM sample_relation
                WHERE
                    episode = {}
                    AND timestep = {}
            ) AS filtered_sample
            JOIN sample
        """.format(episode, timestep))
        return self._main_cursor.fetchone()

    def sample_transitions(self, batch_size):
        self._main_cursor.execute("""
            SELECT curr_obs, action, reward, done FROM sample ORDER BY RANDOM() LIMIT {};
        """.format(batch_size))
        return self._main_cursor.fetchall()

    def get_episode(self, episode):
        self._main_cursor.execute("""
            SELECT curr_obs, action, reward, done
            FROM (
                SELECT sample_id
                FROM sample_relation
                WHERE
                    episode = {}
            ) AS filtered_sample
            JOIN sample
            ON filtered_sample.sample_id = sample.sample_id
        """.format(episode))

        transitions = self._main_cursor.fetchall()

        self._main_cursor.execute("""
            SELECT last_obs
            FROM last_obs
            WHERE
                episode = {}
        """.format(episode))
        last_obs = self._main_cursor.fetchone()[0]

        # Episode is a list of lists of (observations, actions, rewards, dones)
        episode = tuple(map(list, zip(*transitions)))
        episode[0].append(last_obs)
        return episode

    def get_num_transitions(self):
        self._main_cursor.execute("""
            SELECT count(*)
            FROM sample
        """)

        return self._main_cursor.fetchone()[0]

    def get_num_episodes(self):
        self._main_cursor.execute("""
            SELECT max(episode), count(episode)
            FROM last_obs
        """)

        res = self._main_cursor.fetchone()
        assert (res[0] + 1) == res[1], res

        return res[1]

    def close(self):
        self._main_conn.close()
        self._stop_event.set()

if __name__ == "__main__":
    from pprint import pprint
    storage = Storage("test.db")
    
    storage.start()
    for i in range(10):
        storage.save_transition(1, i, np.random.normal(size=(10, 10)), np.random.normal(size=(5,)), np.random.normal(), i == 9, np.random.normal(size=(10, 10)))
    storage.close()

    # You can read at the same time, but this ensures the writes are flushed already
    storage = Storage("test.db")

    row = storage.get_transition(1, 1)
    pprint(row)

    rows = storage.sample_transitions(2)
    for row in rows:
        pprint(row)

    trajectory = storage.get_episode(1)

    print(storage.get_num_transitions())
    for i in trajectory:
        print(len(i))

    print(storage.get_num_episodes())
