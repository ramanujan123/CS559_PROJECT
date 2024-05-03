from .lib.dequedict import DequeDict
from .lib.heapdict import HeapDict
from .lib.pollutionhandler import Pollutionator
from .lib.visualizer import Visualizinator
from .lib.optargs import process_kwargs
from .lib.cachehandler import CacheOp
import numpy as np


class LeCaR:

    # Entry to track the page information
    class LeCaR_Entry:
        def __init__(self, data_block, freq=1, time=0):
            self.data_block = data_block
            self.freq = freq
            self.time = time
            self.evicted_time = None

        # Minimal comparators needed for HeapDict
        def __lt__(self, other):
            if self.freq == other.freq:
                return self.data_block < other.data_block
            return self.freq < other.freq

        # Useful for debugging
        def __repr__(self):
            return "(d={}, f={}, t={})".format(self.data_block, self.freq,
                                               self.time)

    def __init__(self, cache_size, window_size, **kwargs):
        # Randomness and Time
        np.random.seed(123)
        self.time = 0

        # Cache
        self.cache_size = cache_size
        self.lru = DequeDict()
        self.lfu = HeapDict()

        # Histories
        self.history_size = cache_size // 2
        self.lru_hist = DequeDict()
        self.lfu_hist = DequeDict()

        # Decision Weights Initialized
        self.initial_weight = 0.5

        # Fixed Learning Rate
        self.learning_rate = 0.45

        # Fixed Discount Rate
        self.discount_rate = 0.005**(1 / self.cache_size)

        process_kwargs(
            self,
            kwargs,
            acceptable_kws=['learning_rate', 'initial_weight', 'history_size'])

        # Decision Weights
        self.W = np.array([self.initial_weight, 1 - self.initial_weight],
                          dtype=np.float32)
        # Visualize
        self.visualizer = Visualizinator(labels=['W_lru', 'W_lfu', 'hit-rate'],
                                         windowed_labels=['hit-rate'],
                                         window_size=window_size,
                                         **kwargs)

        # Pollution
        self.pollution_handler = Pollutionator(cache_size, **kwargs)

    # True if data_block is in cache (which LRU can represent)
    def __contains__(self, data_block):
        return data_block in self.lru

    def cache_full(self):
        return len(self.lru) == self.cache_size

    # Add Entry to cache with given frequency
    def add_to_cache(self, data_block, freq):
        x = self.LeCaR_Entry(data_block, freq, self.time)
        self.lru[data_block] = x
        self.lfu[data_block] = x

    # Add Entry to history dictated by policy
    def add_to_history(self, x, policy):
        policy_history = None
        if policy == 0:
            policy_history = self.lru_hist
        elif policy == 1:
            policy_history = self.lfu_hist
        elif policy == -1:
            return

        # Evict from history is it is full
        if len(policy_history) == self.history_size:
            evicted = self.get_lru(policy_history)
            del policy_history[evicted.data_block]
        policy_history[x.data_block] = x

    # Get the LRU item in the given DequeDict
    def get_lru(self, deque_dict):
        return deque_dict.first()

    # Get the LFU min item in the LFU (HeapDict)
    def get_heap_min(self):
        return self.lfu.min()

    # Get the random eviction choice based on current weights
    def get_choice(self):
        return 0 if np.random.rand() < self.W[0] else 1

    # Evict an entry
    def evict(self):
        lru = self.get_lru(self.lru)
        lfu = self.get_heap_min()

        evicted = lru
        policy = self.get_choice()

        # Since we're using Entry references, we use is to check
        # that the LRU and LFU Entries are the same Entry
        if lru is lfu:
            evicted, policy = lru, -1
        elif policy == 0:
            evicted = lru
        else:
            evicted = lfu

        del self.lru[evicted.data_block]
        del self.lfu[evicted.data_block]

        evicted.evicted_time = self.time
        self.pollution_handler.remove(evicted.data_block)

        self.add_to_history(evicted, policy)

        return evicted.data_block, policy

    # Cache Hit
    def hit(self, data_block):
        x = self.lru[data_block]
        x.time = self.time

        self.lru[data_block] = x

        x.freq += 1
        self.lfu[data_block] = x

    # Adjust the weights based on the given rewards for LRU and LFU
    def adjust_weights(self, reward_lru, reward_lfu):
        reward = np.array([reward_lru, reward_lfu], dtype=np.float32)
        self.W = self.W * np.exp(self.learning_rate * reward)
        self.W = self.W / np.sum(self.W)

        if self.W[0] >= 0.99:
            self.W = np.array([0.99, 0.01], dtype=np.float32)
        elif self.W[1] >= 0.99:
            self.W = np.array([0.01, 0.99], dtype=np.float32)

    # Cache Miss
    def miss(self, data_block):
        evicted = None

        freq = 1
        if data_block in self.lru_hist:
            entry = self.lru_hist[data_block]
            freq = entry.freq + 1
            del self.lru_hist[data_block]
            reward_lru = -(self.discount_rate
                           **(self.time - entry.evicted_time))
            self.adjust_weights(reward_lru, 0)
        elif data_block in self.lfu_hist:
            entry = self.lfu_hist[data_block]
            freq = entry.freq + 1
            del self.lfu_hist[data_block]
            reward_lfu = -(self.discount_rate
                           **(self.time - entry.evicted_time))
            self.adjust_weights(0, reward_lfu)

        # If the cache is full, evict
        if len(self.lru) == self.cache_size:
            evicted, policy = self.evict()

        self.add_to_cache(data_block, freq)

        return evicted

    # Process and access request for the given data_block
    def request(self, data_block, ts):
        miss = True
        evicted = None
        op = CacheOp.INSERT

        self.time += 1

        self.visualizer.add({
            'W_lru': (self.time, self.W[0], ts),
            'W_lfu': (self.time, self.W[1], ts)
        })

        if data_block in self:
            miss = False
            op = CacheOp.HIT
            self.hit(data_block)
        else:
            evicted = self.miss(data_block)

        # Windowed
        self.visualizer.add_window({'hit-rate': 0 if miss else 1}, self.time, ts)

        # Pollution
        if miss:
            self.pollution_handler.increment_unique_count()
        self.pollution_handler.set_unique(data_block)
        self.pollution_handler.update(self.time)

        return op, evicted
