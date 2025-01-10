import json
from graph import *
import os
import datetime
import warnings


class Cache:
    '''
    Cache for recording the performance of optimizer programs. It serves as two purposes:
    1. Avoid evaluating functionally the same program redundantly. (By using == operator w.r.t. Refined graph)
    2. Record the performance of each program.
    It is designed to read from disk in the beginning of an episode and write the new items back at the end of an episode.
    '''

    def __init__(self, filename='cache.json'):
        self.exist_items = []  # List of items read from disk
        self.new_items = []  # List of items generated in this episode, will be written to disk
        self.filename = filename
        self.invalid_items_count = 0

        # Load items from disk
        with open(self.filename, 'r') as f:
            for line in f:
                try:
                    item = json.loads(line)
                    # Convert update_rule text to a ComputationalGraph object
                    item['graph'] = ComputationalGraph(
                        text_to_update_rule(item.pop('update_rule')))
                    # Check if the graph is valid
                    assert item['graph'].valid, "Invalid graph loaded from disk"
                    item['graph'].refine()
                    self.exist_items.append(item)
                except:
                    # Count invalid items which throw exceptions
                    self.invalid_items_count += 1

        # Print the number of loaded and invalid items in one line
        print(f"Loaded {len(self.exist_items)} items from disk. {self.invalid_items_count} items are invalid.")

    def add_item(self, graph, metric):  # The graph should be the refined version
        assert graph.valid, "Try to add an invalid graph to cache"

        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        pid = os.getpid()

        item = {
            'id': len(self.new_items),
            'graph': graph,
            'timestamp': timestamp,
            'pid': pid,
            'metric': metric,
        }
        self.new_items.append(item)

    def save_to_disk(self, filename=None, clean=False):
        '''
        Save the newly generated items to the disk. It is called in two situations:
        1. At the end of an episode, to save the newly encountered program and its performance in this episode.
        2. When the cache is cleaned, to save the unique items (to another file, in which case you should specify the `filename` parameter).
        '''
        if filename is None:
            filename = self.filename

        # In the 1st situation:
        # The read process is done at the beginning of an episode, and the write is at the end
        # The number of items in the cache file may change during an episode, as other workers may write info
        # So we need to count the number of lines in the file AGAIN to get the next id
        new_id = 0 if clean else self._get_next_id()

        with open(filename, 'a') as f:
            for item in self.new_items:
                item['id'] = new_id
                new_id += 1

                # Convert ComputationalGraph back to string before saving
                item['update_rule'] = show_update_rule(
                    graph_to_update_rule(item.pop('graph')))

                json.dump(item, f)
                f.write('\n')

        print(f"Saved {len(self.new_items)} new items to disk.")

    def find_item(self, graph):
        # The locality principle: check the new items first, with the reversed order
        # Check in new items
        for item in reversed(self.new_items):
            if item['graph'] == graph:
                return True, 'new_items', item['id'], item['metric']

        # Check in existing items
        for item in reversed(self.exist_items):
            if item['graph'] == graph:
                return True, 'exist_items', item['id'], item['metric']

        return False, None, None, None

    def clean_cache(self, filename='cleaned_cache.json'):
        '''
        Remove duplicate items in the cache file.
        It defaults write to another file 'cleaned_cache.json'. Since it calls the save_to_disk() method with append mode, you should make sure that the 'cleaned_cache.json' is empty or nonexistent before calling this method.
        So if you want to move the cleaned cache to the original cache file, you should do it by manually copying, instead of setting the filename parameter to the original cache file.
        '''
        # Check if the file exists and is not empty
        if os.path.isfile(filename) and os.path.getsize(filename) > 0:
            warnings.warn(
                f"Warning: file '{filename}' already exists and is not empty. Please ensure the file is empty or nonexistent before calling this method.")

        # Initialize an empty list to keep the unique items
        unique_items = []

        # Traverse the exist_items list in reversed order
        for item in reversed(self.exist_items):
            graph = item['graph']

            # Check if graph is already in our unique_items list
            is_duplicate = any(
                graph == unique_item['graph'] for unique_item in unique_items)

            # If it's not a duplicate, add it to our unique_items
            if not is_duplicate:
                unique_items.append(item)

        # Replace new_items with the unique items we've found
        self.new_items = unique_items[::-1]

        # Save the cleaned cache to another file in the disk
        self.save_to_disk(filename, clean=True)

        print(
            f"Cache cleaned. {len(self.exist_items) - len(unique_items)} items removed. {len(self.new_items)} unique items saved.")

    def _get_next_id(self):
        i = -1  # Initialize i to -1, so that if the file is empty, i + 1 will return 0
        try:
            with open(self.filename, 'r') as f:
                for i, _ in enumerate(f):
                    pass
            return i + 1
        except FileNotFoundError:
            return 0

# Util Functions
def merge_and_reindex(json_file1, json_file2, output_file):
    '''
    Combine two caches from two separate JSON files into a single cache with reindexed ids.
    '''
    def read_json(file_path):
        data = []
        with open(file_path, 'r') as file:
            for line in file:
                try:
                    data.append(json.loads(line))
                except json.JSONDecodeError:
                    continue  # This skips lines that are not valid JSON
        return data

    def write_json(data, file_path):
        with open(file_path, 'w') as file:
            for item in data:
                file.write(json.dumps(item) + '\n')  # Writes each JSON object on a new line
    # Read data from both files
    data1 = read_json(json_file1)
    data2 = read_json(json_file2)

    # Merge the data
    merged_data = data1 + data2

    # Reindex the ids
    for index, item in enumerate(merged_data):
        item['id'] = index

    # Write the data to a new JSON file
    write_json(merged_data, output_file)


if __name__ == '__main__':
    # Given two cache files from different machine, first merge them into one file
    merge_and_reindex('cache.json', 'cache_byte.json', 'merged_cache.json')
    # Then, clean the merged cache file
    cache = Cache('merged_cache.json')
    cache.clean_cache(filename='cleaned_cache.json')

