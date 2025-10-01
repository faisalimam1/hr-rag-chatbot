"""
Simple LRU cache for query results.
"""
from collections import OrderedDict
import time

class LRUCache:
    def __init__(self, max_size=1024):
        self.max_size = max_size
        self._d = OrderedDict()

    def get(self, key):
        v = self._d.get(key)
        if v is None:
            return None
        # move to end
        self._d.move_to_end(key)
        return v

    def set(self, key, value):
        self._d[key] = value
        self._d.move_to_end(key)
        if len(self._d) > self.max_size:
            self._d.popitem(last=False)

    def __contains__(self, key):
        return key in self._d
