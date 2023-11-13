import copy
import datetime
import json
import os
import re
import sys
import warnings
from typing import List
import chromadb
from chromadb.config import Settings


def get_time() -> tuple:
    """
    Get the current time as a string.

    :return: A tuple containing two strings: the current time and the current UTC time.
    :rtype: tuple
    """
    return (
        f"On {datetime.datetime.now().strftime('%A, %B %-d, %Y at %-I:%M %p')}: ",
        str(datetime.datetime.utcnow()),
    )


def _strip_entry(entry: dict):
    """
    Remove all fields from the entry dictionary except 'role' and 'content'.

    :param entry: A dictionary containing a conversation entry.
    :type entry: dict
    :return: A new dictionary containing only the 'role' and 'content' fields from the original entry.
    :rtype: dict
    """
    if isinstance(entry, list):
        new = []
        for el in entry:
            new.append(_strip_entry(el))
        return new
    else:
        if "name" in entry:
            return {"role": entry["role"], "content": entry["content"], "name": entry["name"]}
        return {"role": entry["role"], "content": entry["content"]}


class AssistantHistory:
    """
    A class to manage the Assistant's conversation history, including storing, reducing,
    and gathering conversation context for future queries.

    :param username: The username of the current user.
    :type username: str
    :param system: The system message.
    :type system: str
    :param tokenizer: A function to tokenize a string.
    :type tokenizer: function
    :param summarizer: A function to summarize a string.
    :type summarizer: function
    :param max_tokens: The maximum number of tokens allowed for the conversation history.
    :type max_tokens: int
    :param embedder: A function to embed a string.
    :type embedder: function, optional
    :param persist_directory: The directory to store the database in.
    :type persist_directory: str, optional
    :param model_injection: Whether to inject the model name into the history.
    :type model_injection: bool, optional
    :param time_injection: Whether to inject the time into the history.
    :type time_injection: bool, optional
    """

    def __init__(
        self,
        username: str,
        system: str,
        tokenizer: callable,
        summarizer: callable,
        max_tokens: int,
        summary_max_tokens: int,
        embedder: callable = None,
        persist_directory: str = "database",
        model_injection: bool = True,
        time_injection: bool = True,
    ):
        """
        Initialize an instance of AssistantHistory.
        """
        self.model_injection = model_injection
        self.time_injection = time_injection
        self.persist_directory = persist_directory
        if getattr(sys, 'frozen', False):
            self.persist_directory = os.path.join(sys._MEIPASS, self.persist_directory)
        self.client = chromadb.PersistentClient(path=self.persist_directory,
                                                settings=Settings(anonymized_telemetry=False))
        # Load metadata from disk, or create a new metadata file if it doesn't exist.
        if os.path.exists(os.path.join(self.persist_directory, "AssistantHistoryMetadata.json")):
            with open(os.path.join(self.persist_directory, "AssistantHistoryMetadata.json"), "r") as f:
                metadata = json.load(f)
        else:
            metadata = {"current_id": "0", "current_summary_id": "0", "to_summarize": []}
        self.next_id = int(metadata["current_id"]) + 1
        self.next_summary_id = int(metadata["current_summary_id"]) + 1
        self.to_summarize = metadata["to_summarize"]

        # Load long-term memory from disk, or create a new LTM file if it doesn't exist.
        if os.path.exists(os.path.join(self.persist_directory, "AssistantHistoryLTM.json")):
            with open(os.path.join(self.persist_directory, "AssistantHistoryLTM.json"), "r") as f:
                ltm = json.load(f)
        else:
            ltm = {"long_term_memory": ""}
        self.long_term_memory = ltm["long_term_memory"]

        # Set instance variables based on input parameters.
        self.username = username
        self.fixed_user = username + "'" if username[-1] == "s" else username + "'s"
        self.system_raw = system
        self.embedder = embedder
        self.max_tokens = max_tokens
        self.summary_max_tokens = summary_max_tokens
        self.tokenizer = tokenizer
        self.summarizer = summarizer
        self.last_context = None
        # Create history and summaries collections using the chromadb client.
        if self.embedder:
            self.history = self.client.get_or_create_collection(
                name="history", embedding_function=self.embedder
            )
            self.summaries = self.client.get_or_create_collection(
                name="summaries", embedding_function=self.embedder
            )
        else:
            self.history = self.client.get_or_create_collection(name="history")
            self.summaries = self.client.get_or_create_collection(name="summaries")

        if self.next_id > 1:
            # Check whether the next summary ID is in the history database.
            last_batch = self.get_history_from_last_batch()
            last_batch_id = last_batch[0]["batch_id"]
            last_summary = self.get_summary_from_id_and_earlier(n_results=1)
            refresh = False
            if len(last_summary) > 0:
                last_summary_batch_id = last_summary[0]["batch_id"]
                if last_summary_batch_id < last_batch_id:
                    not_summarized = self.get_batches(list(range(int(last_summary_batch_id) + 1, int(last_batch_id))))
                    refresh = True
            else:
                not_summarized = last_batch
                refresh = True
            if refresh:
                self.to_summarize += not_summarized
                self.reduce_context()
        self.save_metadata()

    def reload_from_disk(self):
        """
        Reload the Assistant's conversation history from disk.

        :return: None
                """
        self.client = chromadb.PersistentClient(path=self.persist_directory,
                                                settings=Settings(anonymized_telemetry=False))
        if os.path.exists(os.path.join(self.persist_directory, "AssistantHistoryMetadata.json")):
            with open(os.path.join(self.persist_directory, "AssistantHistoryMetadata.json"), "r") as f:
                metadata = json.load(f)
        else:
            metadata = {"current_id": "0", "current_summary_id": "0", "to_summarize": None}
        self.next_id = int(metadata["current_id"]) + 1
        self.next_summary_id = int(metadata["current_summary_id"]) + 1
        self.to_summarize = metadata["to_summarize"]

        if os.path.exists(os.path.join(self.persist_directory, "AssistantHistoryLTM.json")):
            with open(os.path.join(self.persist_directory, "AssistantHistoryLTM.json"), "r") as f:
                ltm = json.load(f)
        else:
            ltm = {"long_term_memory": ""}
        self.long_term_memory = ltm["long_term_memory"]

        if self.embedder:
            self.history = self.client.get_or_create_collection(name="history", embedding_function=self.embedder)
            self.summaries = self.client.get_or_create_collection(name="summaries", embedding_function=self.embedder)
        else:
            self.history = self.client.get_or_create_collection(name="history")
            self.summaries = self.client.get_or_create_collection(name="summaries")
        self.current_user_query = None

    def create_chat_id(self):
        """
        Get the current set ID.

        :return: The current set ID.
        :rtype: str
        """
        new_id = self.next_id
        self.next_id = new_id + 1
        self.save_metadata()
        return str(new_id)

    def create_summary_id(self):
        """
        Get the current summary ID.

        :return: The current summary ID.
        :rtype: str
        """
        new_id = self.next_summary_id
        self.next_summary_id = new_id + 1
        self.save_metadata()
        return str(new_id)

    def save_ltm(self):
        """
        Save the long term memory to a file.
        """
        with open(os.path.join(self.persist_directory, "AssistantHistoryLTM.json"), "w") as f:
            json.dump({"long_term_memory": self.long_term_memory}, f)

    def update_ltm(self):
        self.long_term_memory = self.summarizer(
            self.gather_context("", only_summaries=True,
                                max_tokens=(self.summary_max_tokens -
                                            self.count_tokens_text(self.long_term_memory))))["content"]
        self.save_ltm()

    def count_tokens_text(self, text: str) -> int:
        """
        Count the number of tokens in the given text.

        :param text: A string containing the text to count tokens for.
        :type text: str
        :return: The number of tokens in the given text.
        :rtype: int
        """
        return len(self.tokenizer(text))

    def count_tokens_context(self, ls: list) -> int:
        """
        Count the total number of tokens in a list of conversation entries.

        :param ls: A list of conversation entries.
        :type ls: list
        :return: The total number of tokens in the given list of entries.
        :rtype: int
        """
        total = 0
        for el in ls:
            if isinstance(el, list):
                for e in el:
                    total += self.count_tokens_text(e['content'])
            else:
                total += self.count_tokens_text(el['content'])
        return total

    def add_context(self, context: list) -> None:
        """
        Add query and response to the conversation history.

        :param responses: The assistant responses to be added.
        :type responses: list
        """
        time_str, utc_time = get_time()
        documents = []
        first_id = None
        ids = []
        for i in range(len(context)):
            context[i]['id'] = self.create_chat_id()
            ids.append(context[i]['id'])
            if first_id is None:
                first_id = context[i]['id']
            context[i]['utc_time'] = utc_time
            context[i]['batch_id'] = first_id
            if context[i]['role'] == 'assistant' and context[i].get('model', None) is not None:
                context[i]['model'] = context[i]['model']
            context[i]['num_tokens'] = self.count_tokens_text(context[i]['content'])
            if self.time_injection:
                context[i]['content'] = time_str + context[i]['content']
            if self.model_injection and context[i].get('model', None) is not None:
                context[i]['content'] = "Source AI Model: " + context[i]['model'] + " - " + context[i]['content']
            documents.append(context[i]['content'])

        metadata = copy.deepcopy(context)
        ebeddings = []
        for i in range(len(metadata)):
            ebeddings.append(self.embedder(metadata[i]['content']))
            del metadata[i]['content']

        self.history.add(
            embeddings=ebeddings,
            metadatas=metadata,
            documents=documents,
            ids=ids,
        )
        self.to_summarize + context
        self.save_metadata()

    def get_system(self) -> dict:
        """
        Generate a system message containing user's AI Assistant's name and the current date time.

        :return: A dictionary containing the role and content of the system message.
        :rtype: dict
        """
        system_raw = self.system_raw
        system = re.sub("FIXED_USER_INJECTION", self.fixed_user, system_raw)
        system = re.sub("DATETIME_INJECTION", datetime.datetime.now().strftime("%A, %B %d, %Y at %I:%M %p"), system)
        system = re.sub("LONG_TERM_MEMORY_INJECTION", self.long_term_memory, system)
        return {"role": "system", "content": system}

    def reduce_context(self) -> None:
        """
        Reduce the conversation history by summarizing it.
        """
        if self.to_summarize is None:
            return
        if len(self.to_summarize) == 0:
            return
        batch_ids = []
        for elem in self.to_summarize:
            batch_ids.append(elem['batch_id'])
        for batch in batch_ids:
            source_ids = []
            batch_entries = []
            for el in self.to_summarize:
                if el['batch_id'] == batch:
                    source_ids.append(el['id'])
                    batch_entries.append(el)
            to_reduce = _strip_entry(batch_entries)
            new_summary = self.summarizer(to_reduce)
            new_summary_metadata = {"role": "assistant",
                                    "source_ids": ",".join(source_ids),
                                    "batch_id": batch,
                                    "utc_time": batch_entries[0]['utc_time'],
                                    "num_tokens": self.count_tokens_text(new_summary['content'])}
            self.summaries.add(
                embeddings=[self.embedder(new_summary['content'])],
                metadatas=[new_summary_metadata.copy()],
                documents=[new_summary['content']],
                ids=[self.create_summary_id()],
            )
        self.update_ltm()
        self.to_summarize = []
        self.save_metadata()

    def load_metadata(self):
        """
        Load metadata from the JSON file.
        """
        try:
            with open(os.path.join(self.persist_directory, "AssistantHistoryMetadata.json"), "r") as f:
                data = json.load(f)
                self.next_id = data["current_id"]+1
                self.next_summary_id = data["current_summary_id"]+1
                if self.next_id > 1:
                    self.history.load(os.path.join(self.persist_directory, "AssistantHistory"),
                                      embeddings_storage=os.path.join(self.persist_directory, "embeddings"))
                    self.summaries.load(os.path.join(self.persist_directory, "AssistantSummaries"),
                                        embeddings_storage=os.path.join(self.persist_directory, "embeddings"))
        except FileNotFoundError:
            pass

    def save_metadata(self):
        """
        Save metadata to the JSON file.
        """
        with open(os.path.join(self.persist_directory, "AssistantHistoryMetadata.json"), "w") as f:
            json.dump({"current_id": self.next_id-1, "current_summary_id": self.next_summary_id-1,
                       "to_summarize": self.to_summarize}, f)

    def gather_context(self, query: str or list, minimum_recent_history_length: int = 2, max_tokens: int = None,
                       only_summaries: bool = False, only_necessary_fields: bool = True,
                       distance_cut_off: float = None, query_n_max: int = 3, verbose: bool = False) -> List[dict]:
        """
        Gathers relevant context for a given query from the chat assistant's history.

        :param query: The input query for which context is to be gathered
        :type query: str or list
        :param minimum_recent_history_length: The minimum number of recent history entries to include, defaults to 2
        :type minimum_recent_history_length: int, optional
        :param max_tokens: The maximum number of tokens allowed in the combined context, defaults to 2500
        :type max_tokens: int, optional
        :param only_summaries: Whether to only include summaries in the context, defaults to False
        :type only_summaries: bool, optional
        :param only_necessary_fields: Whether to only include 'role' and 'content' in context entries, defaults to True
        :type only_necessary_fields: bool, optional
        :param distance_cut_off: The maximum distance between the query and the context entry, defaults to None
        :type distance_cut_off: float, optional
        :param query_n_max: The maximum number results to return from full context query, defaults to 30
        :type query_n_max: int, optional
        :param verbose: Whether to print out context info
        :return: A list of relevant context entries for the given query
        :rtype: List[dict]
        """
        info = {"recent_context":{"ids": [], "num_tokens": 0},
                "history_query": {"ids": [], "num_tokens": 0},
                "summaries_queries": {"ids": [], "num_tokens": 0},
                "summaries": {"ids": [], "num_tokens": 0}}
        if isinstance(query, list):
            query_str = ""
            for el in query:
                query_str = query_str + "\n\n" + el["content"]
            query = query_str

        if max_tokens is None:
            max_tokens = int(0.85 * self.max_tokens)
        if not only_summaries:
            context_list = []
            id_added = []
            summary_ids = []
            token_count = self.count_tokens_text(query)

            # Add the most recent history entries
            recent_history_length = 0
            while recent_history_length < minimum_recent_history_length:
                if recent_history_length == 0:
                    batch_context = self.get_history_from_last_batch()
                    if batch_context is None:
                        break
                else:
                    batch_context = self.get_batch_before(batch_id)
                    if batch_context is None:
                        break
                batch_id = batch_context[0]["batch_id"]
                batch_context.reverse()
                for entry in batch_context:
                    if token_count + entry["num_tokens"] > max_tokens:
                        break
                    context_list.insert(0, entry)
                    tid = entry["id"]
                    id_added.append(tid)
                    token_count += entry["num_tokens"]
                    info["recent_context"]["ids"].append(tid)
                    info["recent_context"]["num_tokens"] += entry["num_tokens"]
                recent_history_length += 1


            # If the context is too short, query the full history
            if token_count < max_tokens and self.next_id > 1:
                query_size = query_n_max
                if query_size > self.next_id - 1:
                    query_size = self.next_id - 1
                query_results = self.history.query(query_embeddings=[self.embedder(query)], n_results=query_size)
                for id in query_results["ids"][0][:query_size]:
                    if id not in id_added:
                        result_pos = query_results["ids"][0].index(id)
                        entry = query_results["metadatas"][0][result_pos]
                        if token_count + entry["num_tokens"] > max_tokens:
                            break
                        if distance_cut_off is not None:
                            if query_results["distances"][0][result_pos] < distance_cut_off:
                                break
                        batch_data = self.get_batches(entry["batch_id"])
                        batch_data.reverse()
                        for item in batch_data:
                            if token_count + item["num_tokens"] > max_tokens:
                                break
                            if item['id'] not in id_added:
                                context_list.insert(0, item)
                                token_count += item["num_tokens"]
                                id_added.append(item['id'])
                                info["history_query"]["ids"].append(item['id'])
                                info["history_query"]["num_tokens"] += item["num_tokens"]

                # If the context is still too short, query the summaries
                if token_count < max_tokens:
                    query_size = query_n_max
                    if query_size > self.next_summary_id - 1:
                        query_size = self.next_summary_id - 1
                    query_summaries = self.summaries.query(query_embeddings=[self.embedder(query)], n_results=query_size)
                    for id in query_summaries["ids"][0]:
                        if id not in summary_ids:
                            result_pos = query_summaries["ids"][0].index(id)
                            entry = query_summaries["metadatas"][0][result_pos]
                            if token_count + entry["num_tokens"] > max_tokens:
                                break
                            if distance_cut_off is not None:
                                if query_summaries["distances"][0][result_pos] < distance_cut_off:
                                    break
                            if not (entry["source_ids"].split(",")[0] in id_added and
                                    entry["source_ids"].split(",")[1] in id_added):
                                entry["content"] = query_summaries["documents"][0][result_pos]
                                summary_ids.append(id)
                                context_list.insert(0, entry)
                                token_count += entry["num_tokens"]
                                info["summaries_queries"]["ids"].append(id)


            else:
                context_list = []
                summary_ids = []
                id_added = []
                token_count = self.count_tokens_text(query)
        else:
            context_list = []
            summary_ids = []
            id_added = []
            token_count = self.count_tokens_text(query)

        # Add the summaries if there is any space left
        current_summary_id = self.next_summary_id - 1
        while current_summary_id > 0 and token_count < max_tokens:
            if current_summary_id not in summary_ids:
                result = self.summaries.get(ids=str(current_summary_id), include=['documents', 'metadatas'])
                entry = result["metadatas"][0]
                entry["content"] = result["documents"][0]
                if token_count + entry["num_tokens"] > max_tokens:
                    break
                if not (entry["source_ids"].split(",")[0] in id_added and
                        entry["source_ids"].split(",")[1] in id_added):
                    context_list.insert(0, entry)
                    summary_ids.append(result["ids"][0])
                    token_count += entry["num_tokens"]
                    info["summaries"]["ids"].append(result["ids"][0])
                    info["summaries"]["num_tokens"] += entry["num_tokens"]
            current_summary_id -= 1

        assert token_count <= max_tokens

        if only_necessary_fields:
            context_list = [_strip_entry(entry) for entry in context_list]

        self.last_context = [self.get_system()] + context_list

        if verbose:
            from pprint import pprint
            pprint(info)

        return self.last_context

    def get_history(self):
        """
        Returns the history of the chat assistant

        :return: The complete history
        :rtype: list
        """
        return self.history

    def get_history_from_id_and_earlier(self, id=None, n_results=10, reload_disk=False):
        """
        Returns the history of the chat assistant from a given id and earlier

        :param id: The id to start from
        :type id: int
        :param n_results: The number of results to return
        :type n_results: int
        :param reload_disk: If the history should be reloaded from disk
        :type reload_disk: bool
        :return: The history
        :rtype: list
        """
        if reload_disk:
            self.reload_from_disk()
        if id is None:
            id = self.next_id-1
        else:
            id = int(id)
        target_ids = list(range(id, id-n_results, -1))
        target_ids = [str(x) for x in target_ids if x > 0]
        output = []
        results = self.history.get(ids=target_ids, include=['documents', 'metadatas'])
        for tid in target_ids:
            if tid in results["ids"]:
                pos = results["ids"].index(tid)
                entry = results["metadatas"][pos]
                entry["content"] = results["documents"][pos]
                entry["id"] = tid
                output.append(entry)
        return output

    def get_history_from_last_batch(self):
        """
        Returns the history of the chat assistant from the last batch

        :return: The history
        :rtype: list
        """
        tid = self.next_id-1
        results = self.history.get(ids=str(tid), include=['documents', 'metadatas'])
        if str(tid) in results["ids"]:
            pos = results["ids"].index(str(tid))
            entry = results["metadatas"][pos]
            entry["content"] = results["documents"][pos]
            entry["id"] = tid
            bid = entry["batch_id"]
            return self.get_batches([bid])
        return None

    def get_batch_before(self, bid):
        """
        Returns the history of the chat assistant from the last batch

        :return: The history
        :rtype: list
        """
        current_batch = self.get_batches([bid])
        lowest_id = None
        for entry in current_batch:
            tid = int(entry["id"])
            if lowest_id is None:
                lowest_id = tid
            if tid < lowest_id:
                lowest_id = tid
        in_next_batch = lowest_id - 1
        if in_next_batch < 0:
            return None
        result = self.get_history_from_id_and_earlier(in_next_batch, 1)
        if result is None:
            return None
        if len(result) == 0:
            return None
        next_batch_id = result[0]["batch_id"]
        return self.get_batches([next_batch_id])

    def get_batches(self, bids):
        """
        Returns the history of the chat assistant from the last batch

        :return: The history
        :rtype: list
        """
        output = []
        for bid in bids:
            results = self.history.get(include=['documents', 'metadatas'], where={"batch_id": str(bid)})
            ids = results["ids"]
            ids.sort()
            for tid in ids:
                pos = results["ids"].index(str(tid))
                entry = results["metadatas"][pos]
                entry["content"] = results["documents"][pos]
                entry["id"] = tid
                output.append(entry)
        return output

    def get_summary_from_id_and_earlier(self, id=None, n_results=10, reload_disk=False):
        """
        Returns the summary of the chat assistant from a given id and earlier

        :param id: The id to start from
        :type id: int
        :param n_results: The number of results to return
        :type n_results: int
        :param reload_disk: If the history should be reloaded from disk
        :type reload_disk: bool
        :return: The history
        :rtype: list
        """
        if reload_disk:
            self.reload_from_disk()
        if id is None:
            id = self.next_summary_id-1
        else:
            id = int(id)
        target_ids = list(range(id, id-n_results, -1))
        target_ids = [str(x) for x in target_ids if x > 0]
        output = []
        results = self.summaries.get(ids=target_ids, include=['documents', 'metadatas'])
        for tid in target_ids:
            if tid in results["ids"]:
                pos = results["ids"].index(str(tid))
                entry = results["metadatas"][pos]
                entry["content"] = results["documents"][pos]
                entry["id"] = tid
                output.append(entry)
        if len(output) > 1:
            warnings.warn("Chat assistant: More than one summary found for id {}".format(id))
        return output
