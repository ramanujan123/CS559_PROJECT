from .lib.dequedict import DequeDict
from .lib.pollutionhandler import Pollutionator
from .lib.visualizer import Visualizinator
from .lib.cachehandler import CacheOp


class LeastRecentlyUsed:
    class LRU_Entry:
        def __init__(self, data_block):
            self.data_block = data_block

        def __repr__(self):
            return "(d={})".format(self.data_block)

    def __init__(self, cache_size, window_size, **kwargs):
        self.cache_size = cache_size
        self.lru = DequeDict()

        self.time = 0

        self.visualizer = Visualizinator(labels=['hit-rate'],
                                         windowed_labels=['hit-rate'],
                                         window_size=window_size,
                                         **kwargs)

        self.pollution_handler = Pollutionator(cache_size, **kwargs)

    def __contains__(self, data_block):
        return data_block in self.lru

    def cache_full(self):
        return len(self.lru) == self.cache_size

    def add_to_cache(self, data_block):
        x = self.LRU_Entry(data_block)
        self.lru[data_block] = x

    def hit(self, data_block):
        x = self.lru[data_block]
        self.lru[data_block] = x

    def evict(self):
        lru = self.lru.pop_first()
        self.pollution_handler.remove(lru.data_block)
        return lru.data_block

    def miss(self, data_block):
        evicted = None

        if len(self.lru) == self.cache_size:
            evicted = self.evict()
        self.add_to_cache(data_block)

        return evicted

    def process_request(self, data_block, ts):
        miss = True
        evicted = None

        self.time += 1

        if data_block in self:
            miss = False
            self.hit(data_block)
        else:
            evicted = self.miss(data_block)

        self.visualizer.add_window({'hit-rate': 0 if miss else 1}, self.time, ts)

        # Pollutionator
        if miss:
            self.pollution_handler.increment_unique_count()
        self.pollution_handler.set_unique(data_block)
        self.pollution_handler.update(self.time)

        op = CacheOp.INSERT if miss else CacheOp.HIT

        return op, evicted
