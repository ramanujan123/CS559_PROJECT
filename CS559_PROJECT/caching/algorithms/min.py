from .lib.pollutionhandler import Pollutionator
from .lib.visualizer import Visualizinator
from .lib.cachehandler import CacheOp

class Minimum:
    class MIN_Entry:
        def __init__(self, data_block, index):
            self.data_block = data_block
            self.index = index

        def __repr__(self):
            return "(d={}, i={})".format(self.data_block, self.index)

    class MIN_index:
        def __init__(self, index, count):
            self.index_ = index
            self.count = count

    def __init__(self, cache_size, window_size, **kwargs):
        self.cache_size = cache_size
        
        # Set of data blocks 
        self.request_data_blocks = {}
        # Set of counters per index block
        self.request_index = {}
        
        # Counters per index
        self.completion_count = 1
        self.current_index = 0

        self.time = 0

        self.visualizer = Visualizinator(labels=['hit-rate'],
                                         windowed_labels=['hit-rate'],
                                         window_size=window_size,
                                         **kwargs)
        self.pollution_handler = Pollutionator(cache_size, **kwargs)

    def increment_counter(self, index):
        if index in self.request_index:
            c = self.request_index[index]
            c.count += 1
        else:
            c = self.MIN_index(index, count=0)
            c.count += 1
            self.request_index[index] = c

    def decrement_counter(self, index):
        if index in self.request_index:
            c = self.request_index[index]
        else:
            return
        c.count -= 1
        if c.count == 0:
            del self.request_index[index]
    
    def get_index_count(self, index):
        if index in self.request_index:
            c = self.request_index[index]
            return c.count
        return 0

    def process_request(self, data_block, ts):
        miss = True
        evicted = None
        self.time += 1

        counter = 0
        temp = 0
        
        if data_block in self.request_data_blocks:
            x = self.request_data_blocks[data_block]
        else:
            x = self.MIN_Entry(data_block, index=0)
            self.request_data_blocks[data_block] = x

        if (x.index < self.completion_count):
            self.current_index += 1
            self.decrement_counter(x.index)
            x.index = self.current_index
            self.increment_counter(x.index)
            
            op = CacheOp.INSERT if miss else CacheOp.HIT
            self.visualizer.add_window({'hit-rate': 0 if miss else 1}, self.time, ts)
            return op, evicted

        if (x.index == self.current_index):
            miss = False
            op = CacheOp.INSERT if miss else CacheOp.HIT
            self.visualizer.add_window({'hit-rate': 0 if miss else 1}, self.time, ts)
            return op, evicted
        
        if (x.index < self.current_index and x.index >= self.completion_count):
            self.decrement_counter(x.index)
            x.index = self.current_index
            self.increment_counter(x.index)
            temp = self.current_index
            counter = 0
            
            while (1):
                counter += self.get_index_count(temp)
                if (counter == self.cache_size or temp == self.completion_count):
                    self.completion_count = temp

                    miss = False
                    op = CacheOp.INSERT if miss else CacheOp.HIT
                    self.visualizer.add_window({'hit-rate': 0 if miss else 1}, self.time, ts)
                    return op, evicted
            
                if counter < self.cache_size:
                    counter -= 1
                    temp -= 1
                    continue
 
            assert(False)
        
        miss = False
        op = CacheOp.INSERT if miss else CacheOp.HIT
        self.visualizer.add_window({'hit-rate': 0 if miss else 1}, self.time, ts)
        return op, evicted
