from .lib.heapdict import HeapDict
from .lib.pollutionhandler import Pollutionator
from .lib.visualizer import Visualizinator
from .lib.cachehandler import CacheOp


class LFU:
    class LFU_Entry:
        def __init__(self, data_block, freq=1, time=0):
            self.data_block = data_block
            self.freq = freq
            self.time = time

        def __lt__(self, other):
            if self.freq == other.freq:
                return self.time > other.time
            return self.freq < other.freq

        def __repr__(self):
            return "(d={}, f={}, t={})".format(self.data_block, self.freq,
                                               self.time)

    def __init__(self, cache_size, window_size, **kwargs):
        self.cache_size = cache_size
        self.lfu = HeapDict()
        self.time = 0
        self.visualizer = Visualizinator(labels=['hit-rate'],
                                         windowed_labels=['hit-rate'],
                                         window_size=window_size,
                                         **kwargs)

        self.pollution_handler = Pollutionator(cache_size, **kwargs)

    def __contains__(self, data_block):
        return data_block in self.lfu

    def cache_full(self):
        return len(self.lfu) == self.cache_size

    def add_to_cache(self, data_block):
        x = self.LFU_Entry(data_block, freq=1, time=self.time)
        self.lfu[data_block] = x

    def hit(self, data_block):
        x = self.lfu[data_block]
        x.freq += 1
        x.time = self.time
        self.lfu[data_block] = x

    def evict(self):
        lfu_min = self.lfu.pop_min()
        self.pollution_handler.remove(lfu_min.data_block)
        return lfu_min.data_block

    def miss(self, data_block):
        evicted = None

        if len(self.lfu) == self.cache_size:
            evicted = self.evict()
        self.add_to_cache(data_block)

        return evicted

    def request(self, data_block, ts):
        miss = True
        evicted = None
        op = CacheOp.INSERT

        self.time += 1

        if data_block in self:
            miss = False
            op = CacheOp.HIT
            self.hit(data_block)
        else:
            evicted = self.miss(data_block)

        self.visualizer.add_window({'hit-rate': 0 if miss else 1}, self.time, ts)

        # Pollutionator
        if miss:
            self.pollution_handler.increment_unique_count()
        self.pollution_handler.set_unique(data_block)
        self.pollution_handler.update(self.time)

        return op, evicted
