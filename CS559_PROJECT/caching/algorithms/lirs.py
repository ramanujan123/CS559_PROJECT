from .lib.dequedict import DequeDict
from .lib.pollutionhandler import Pollutionator
from .lib.optargs import process_kwargs
from .lib.visualizer import Visualizinator
from .lib.cachehandler import CacheOp


class LIRS:
    class LIRS_Entry:
        def __init__(self, data_block, is_LIR=False, in_cache=True):
            self.data_block = data_block
            self.is_LIR = is_LIR
            self.in_cache = in_cache

        def __repr__(self):
            return "(d={}, is_LIR={}, in_cache={})".format(
                self.data_block, self.is_LIR, self.in_cache)

    def __init__(self, cache_size, window_size, **kwargs):
        self.cache_size = cache_size

        self.hirs_ratio = 0.01

        process_kwargs(self, kwargs, acceptable_kws=['hirs_ratio'])

        self.hirs_limit = max(2, int((self.cache_size * self.hirs_ratio)))
        self.lirs_limit = self.cache_size - self.hirs_limit

        self.hirs_count = 0
        self.lirs_count = 0
        self.nonresident = 0

        # s stack, semi-split to find nonresident HIRs quickly
        self.s = DequeDict()
        self.nr_hirs = DequeDict()
        # q, the resident HIR stack
        self.q = DequeDict()

        self.time = 0
        self.last_data_block = None
        self.visualizer = Visualizinator(labels=['hit-rate', 'q_size'],
                                         windowed_labels=['hit-rate'],
                                         window_size=window_size,
                                         **kwargs)
        self.pollution_handler = Pollutionator(cache_size, **kwargs)

    def __contains__(self, data_block):
        if data_block in self.s:
            return self.s[data_block].in_cache
        return data_block in self.q

    def cache_full(self):
        return self.lirs_count + self.hirs_count == self.cache_size

    def hit_LIR(self, data_block):
        lru_lir = self.s.first()
        x = self.s[data_block]
        self.s[data_block] = x
        if lru_lir is x:
            self.prune()

    def prune(self):
        while self.s:
            x = self.s.first()
            if x.is_LIR:
                break

            del self.s[x.data_block]
            if not x.in_cache:
                del self.nr_hirs[x.data_block]
                self.nonresident -= 1

    def hit_HIR_in_LIRS(self, data_block):
        evicted = None

        x = self.s[data_block]

        if x.in_cache:
            del self.s[data_block]
            del self.q[data_block]
            self.hirs_count -= 1
        else:
            del self.s[data_block]
            del self.nr_hirs[data_block]
            self.nonresident -= 1

            if self.cache_full():
                evicted = self.eject_HIR()

        if self.lirs_count >= self.lirs_limit:
            self.eject_LIR()

        self.s[data_block] = x
        x.in_cache = True
        x.is_LIR = True
        self.lirs_count += 1

        return evicted

    def eject_LIR(self):
        assert (self.s.first().is_LIR)

        lru = self.s.pop_first()
        self.lirs_count -= 1
        lru.is_LIR = False

        self.q[lru.data_block] = lru
        self.hirs_count += 1

        self.prune()

    def eject_HIR(self):
        lru = self.q.pop_first()
        self.hirs_count -= 1

        if lru.data_block in self.s:
            self.nr_hirs[lru.data_block] = lru
            lru.in_cache = False
            self.nonresident += 1

        self.pollution_handler.remove(lru.data_block)

        return lru.data_block

    def hit_HIR_in_Q(self, data_block):
        x = self.q[data_block]
        self.q[data_block] = x
        self.s[data_block] = x

    def limit_stack(self):
        while len(self.s) > (2 * self.cache_size):
            lru = self.nr_hirs.pop_first()
            del self.s[lru.data_block]
            self.nonresident -= 1

    def miss(self, data_block):
        evicted = None

        if self.cache_full():
            evicted = self.eject_HIR()

        if self.lirs_count < self.lirs_limit and self.hirs_count == 0:
            x = self.LIRS_Entry(data_block, is_LIR=True)
            self.s[data_block] = x
            self.lirs_count += 1
        else:
            x = self.LIRS_Entry(data_block, is_LIR=False)
            self.s[data_block] = x
            self.q[data_block] = x
            self.hirs_count += 1

        return evicted

    def request(self, data_block, ts):
        miss = data_block not in self
        evicted = None

        self.time += 1
        self.visualizer.add({
            'q_size': (self.time, self.hirs_limit, ts)
        })
        if data_block != self.last_data_block:
            self.last_data_block = data_block

            if data_block in self.s:
                x = self.s[data_block]
                if x.is_LIR:
                    self.hit_LIR(data_block)
                else:
                    evicted = self.hit_HIR_in_LIRS(data_block)
            elif data_block in self.q:
                self.hit_HIR_in_Q(data_block)
            else:
                evicted = self.miss(data_block)

        self.limit_stack()

        self.visualizer.add_window({'hit-rate': 0 if miss else 1}, self.time, ts)

        if miss:
            self.pollution_handler.increment_unique_count()
        self.pollution_handler.set_unique(data_block)
        self.pollution_handler.update(self.time)

        op = CacheOp.INSERT if miss else CacheOp.HIT

        return op, evicted
