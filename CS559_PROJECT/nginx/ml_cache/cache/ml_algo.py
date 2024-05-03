from .lib.dequedict import MyDequeDict
from .lib.heapdict import MyHeapDict
from .lib.pollutionhandler import PollutionHandler
from .lib.visualizer import Visualizer
from .lib.optargs import process_kwargs
from .lib.cachehandler import CacheHandler
import numpy as np

class CustomMLAlgorithm:

    # Inner class for storing cache entries
    class AlgorithmEntry:
        def __init__(self, data_block, frequency=1, timestamp=0, is_new=True):
            self.data_block = data_block
            self.frequency = frequency
            self.timestamp = timestamp
            self.evicted_timestamp = None
            self.is_demoted = False
            self.is_new = is_new

        def __lt__(self, other):
            if self.frequency == other.frequency:
                return self.timestamp > other.timestamp
            return self.frequency < other.frequency

        def __repr__(self):
            return "(d={}, f={}, t={})".format(self.data_block, self.frequency, self.timestamp)

    # Inner class for controlling the adaptive learning rate
    class LearningRateController:
        def __init__(self, period_length, **kwargs):
            self.learning_rate = np.sqrt((2.0 * np.log(2)) / period_length)

            process_kwargs(self, kwargs, acceptable_kws=['learning_rate'])

            self.learning_rate_reset = min(max(self.learning_rate, 0.001), 1)
            self.current_learning_rate = self.learning_rate
            self.previous_learning_rate = 0.0
            self.learning_rates = []

            self.period_length = period_length

            self.hit_rate = 0
            self.previous_hit_rate = 0.0
            self.previous_hit_rate_difference = 0.0
            self.hit_rate_zero_count = 0
            self.hit_rate_negative_count = 0

        def __mul__(self, other):
            return self.learning_rate * other

        # Update the adaptive learning rate
        def update(self, time):
            if time % self.period_length == 0:
                current_hit_rate = round(self.hit_rate / float(self.period_length), 3)
                hit_rate_difference = round(current_hit_rate - self.previous_hit_rate, 3)

                delta_lr = round(self.current_learning_rate, 3) - round(self.previous_learning_rate, 3)
                delta, delta_hr = self.update_in_delta_direction(delta_lr, hit_rate_difference)

                if delta > 0:
                    self.learning_rate = min(self.learning_rate + abs(self.learning_rate * delta_lr), 1)
                    self.hit_rate_negative_count = 0
                    self.hit_rate_zero_count = 0
                elif delta < 0:
                    self.learning_rate = max(self.learning_rate - abs(self.learning_rate * delta_lr), 0.001)
                    self.hit_rate_negative_count = 0
                    self.hit_rate_zero_count = 0
                elif delta == 0 and hit_rate_difference <= 0:
                    if hit_rate_curr <= 0 and hit_rate_difference == 0:
                        self.hit_rate_zero_count += 1
                    if hit_rate_difference < 0:
                        self.hit_rate_negative_count += 1
                        self.hit_rate_zero_count += 1
                    if self.hit_rate_zero_count >= 10:
                        self.learning_rate = self.learning_rate_reset
                        self.hit_rate_zero_count = 0
                    elif hit_rate_difference < 0:
                        if self.hit_rate_negative_count >= 10:
                            self.learning_rate = self.learning_rate_reset
                            self.hit_rate_negative_count = 0
                        else:
                            self.update_in_random_direction()
                self.previous_learning_rate = self.current_learning_rate
                self.current_learning_rate = self.learning_rate
                self.previous_hit_rate = current_hit_rate
                self.previous_hit_rate_difference = hit_rate_difference
                self.hit_rate = 0

            self.learning_rates.append(self.learning_rate)

        # Update the learning rate based on the change in learning_rate and hit_rate
        def update_in_delta_direction(self, learning_rate_difference, hit_rate_difference):
            delta = learning_rate_difference * hit_rate_difference
            delta = int(delta / abs(delta)) if delta != 0 else 0
            delta_hr = 0 if delta == 0 and learning_rate_difference != 0 else 1
            return delta, delta_hr

        # Update the learning rate in a random direction or correct it from extremes
        def update_in_random_direction(self):
            if self.learning_rate >= 1:
                self.learning_rate = 0.9
            elif self.learning_rate <= 0.001:
                self.learning_rate = 0.005
            elif np.random.choice(['Increase', 'Decrease']) == 'Increase':
                self.learning_rate = min(self.learning_rate * 1.25, 1)
            else:
                self.learning_rate = max(self.learning_rate * 0.75, 0.001)

    def __init__(self, cache_size, window_size, **kwargs):
        np.random.seed(123)
        self.time = 0
        self.cache_size = cache_size
        self.stack_s = MyDequeDict()
        self.stack_q = MyDequeDict()
        self.lfu_heap = MyHeapDict()
        self.history_size = cache_size // 2
        self.lru_history = MyDequeDict()
        self.lfu_history = MyDequeDict()
        self.initial_weight = 0.5
        self.learning_rate_controller = self.LearningRateController(cache_size, **kwargs)
        process_kwargs(self, kwargs, acceptable_kws=['initial_weight', 'history_size'])
        self.weights = np.array([self.initial_weight, 1 - self.initial_weight], dtype=np.float32)
        stack_ratio = 0.01
        self.queue_limit = max(1, int((stack_ratio * self.cache_size) + 0.5))
        self.stack_limit = self.cache_size - self.queue_limit
        self.queue_size = 0
        self.stack_size = 0
        self.demoted_count = 0
        self.normal_count = 0
        self.queue_sizes = []
        self.visualizer = Visualizer(labels=['W_lru', 'W_lfu', 'hit-rate', 'queue_size'],
                                     windowed_labels=['hit-rate'],
                                     window_size=window_size,
                                     **kwargs)
        self.pollution_handler = PollutionHandler(cache_size, **kwargs)

    def __contains__(self, data_block):
        return (data_block in self.stack_s or data_block in self.stack_q)

    def cache_full(self):
        return len(self.stack_s) + len(self.stack_q) == self.cache_size

    def hit_in_stack_s(self, data_block):
        x = self.stack_s[data_block]
        x.timestamp = self.time
        self.stack_s[data_block] = x
        x.frequency += 1
        self.lfu_heap[data_block] = x

    def hit_in_stack_q(self, data_block):
        x = self.stack_q[data_block]
        x.timestamp = self.time
        x.frequency += 1
        self.lfu_heap[data_block] = x
        if x.is_demoted:
            self.adjust_size(True)
            x.is_demoted = False
            self.demoted_count -= 1
        del self.stack_q[x.data_block]
        self.queue_size -= 1
        if self.stack_size >= self.stack_limit:
            y = self.stack_s.pop_first()
            y.is_demoted = True
            self.demoted_count += 1
            self.stack_size -= 1
            self.stack_q[y.data_block] = y
            self.queue_size += 1
        self.stack_s[x.data_block] = x
        self.stack_size += 1

    def add_to_stack_s(self, data_block, frequency, is_new=True):
        x = self.AlgorithmEntry(data_block, frequency, self.time, is_new)
        self.stack_s[data_block] = x
        self.lfu_heap[data_block] = x
        self.stack_size += 1

    def add_to_stack_q(self, data_block, frequency, is_new=True):
        x = self.AlgorithmEntry(data_block, frequency, self.time, is_new)
        self.stack_q[data_block] = x
        self.lfu_heap[data_block] = x
        self.queue_size += 1

    def add_to_history(self, x, policy):
        policy_history = None
        if policy == 0:
            policy_history = self.lru_history
            if x.is_new:
                self.normal_count += 1
        elif policy == 1:
            policy_history = self.lfu_history
        elif policy == -1:
            return
        if len(policy_history) == self.history_size:
            evicted = self.get_lru(policy_history)
            del policy_history[evicted.data_block]
            if policy_history == self.lru_history and evicted.is_new:
                self.normal_count -= 1
        policy_history[x.data_block] = x

    def get_lru(self, deque_dict):
        return deque_dict.first()

    def get_heap_min(self):
        return self.lfu_heap.min()

    def get_choice(self):
        return 0 if np.random.rand() < self.weights[0] else 1

    def evict_entry(self):
        lru = self.get_lru(self.stack_q)
        lfu = self.get_heap_min()
        evicted = lru
        policy = self.get_choice()
        if lru is lfu:
            evicted, policy = lru, -1
        elif policy == 0:
            evicted = lru
            del self.stack_q[evicted.data_block]
            self.queue_size -= 1
        elif policy == 1:
            evicted = lfu
            if evicted.data_block in self.stack_s:
                del self.stack_s[evicted.data_block]
                self.stack_size -= 1
            elif evicted.data_block in self.stack_q:
                del self.stack_q[evicted.data_block]
                self.queue_size -= 1
        if evicted.is_demoted:
            self.demoted_count -= 1
            evicted.is_demoted = False
        if policy == -1:
            del self.stack_q[evicted.data_block]
            self.queue_size -= 1
        del self.lfu_heap[evicted.data_block]
        evicted.evicted_timestamp = self.time
        self.pollution_handler.remove(evicted.data_block)
        self.add_to_history(evicted, policy)
        return evicted.data_block, policy

    def adjust_weights(self, reward_lru, reward_lfu):
        reward = np.array([reward_lru, reward_lfu], dtype=np.float32)
        self.weights = self.weights * np.exp(self.learning_rate_controller.current_learning_rate * reward)
        self.weights = self.weights / np.sum(self.weights)
        if self.weights[0] >= 0.99:
            self.weights = np.array([0.99, 0.01], dtype=np.float32)
        elif self.weights[1] >= 0.99:
            self.weights = np.array([0.01, 0.99], dtype=np.float32)

    def adjust_size(self, hit_in_q):
        if hit_in_q:
            demoted_count = max(1, self.demoted_count)
            self.stack_limit = min(self.cache_size - 1, self.stack_limit + max(1, int((self.normal_count / demoted_count) + 0.5)))
            self.queue_limit = self.cache_size - self.stack_limit
        else:
            normal_count = max(1, self.normal_count)
            self.queue_limit = min(self.cache_size - 1, self.queue_limit + max(1, int((self.demoted_count / normal_count) + 0.5))))
            self.stack_limit = self.cache_size - self.queue_limit

    def hit_in_lru_history(self, data_block):
        evicted = None
        entry = self.lru_history[data_block]
        frequency = entry.frequency + 1
        del self.lru_history[data_block]
        if entry.is_new:
            self.normal_count -= 1
            entry.is_new = False
            self.adjust_size(False)
        self.adjust_weights(-1, 0)
        if (self.stack_size + self.queue_size) >= self.cache_size:
            evicted, policy = self.evict_entry()
        self.add_to_stack_s(entry.data_block, entry.frequency, is_new=False)
        self.limit_stack()
        return evicted

    def hit_in_lfu_history(self, data_block):
        evicted = None
        entry = self.lfu_history[data_block]
        frequency = entry.frequency + 1
        del self.lfu_history[data_block]
        self.adjust_weights(0, -1)
        if (self.stack_size + self.queue_size) >= self.cache_size:
            evicted, policy = self.evict_entry()
        self.add_to_stack_s(entry.data_block, entry.frequency, is_new=False)
        self.limit_stack()
        return evicted

    def limit_stack(self):
        while self.stack_size >= self.stack_limit:
            demoted = self.stack_s.pop_first()
            self.stack_size -= 1
            demoted.is_demoted = True
            self.demoted_count += 1
            self.stack_q[demoted.data_block] = demoted
            self.queue_size += 1

    def miss(self, data_block):
        evicted = None
        frequency = 1
        if self.stack_size < self.stack_limit and self.queue_size == 0:
            self.add_to_stack_s(data_block, frequency, is_new=False)
        elif self.stack_size + self.queue_size < self.cache_size and self.queue_size < self.queue_limit:
            self.add_to_stack_q(data_block, frequency, is_new=False)
        else:
            if (self.stack_size + self.queue_size) >= self.cache_size:
                evicted, policy = self.evict_entry()
            self.add_to_stack_q(data_block, frequency, is_new=True)
            self.limit_stack()
        return evicted

    def process_request(self, data_block, timestamp):
        is_miss = False
        evicted = None
        self.time += 1
        self.visualizer.add({'W_lru': (self.time, self.weights[0], timestamp),
                             'W_lfu': (self.time, self.weights[1], timestamp),
                             'queue_size': (self.time, self.queue_size, timestamp)})
        self.learning_rate_controller.update(self.time)
        if data_block in self.stack_s:
            self.hit_in_stack_s(data_block)
        elif data_block in self.stack_q:
            self.hit_in_stack_q(data_block)
        elif data_block in self.lru_history:
            is_miss = True
            evicted = self.hit_in_lru_history(data_block)
        elif data_block in self.lfu_history:
            is_miss = True
            evicted = self.hit_in_lfu_history(data_block)
        else:
            is_miss = True
            evicted = self.miss(data_block)
        self.visualizer.add_window({'hit-rate': 0 if is_miss else 1}, self.time, timestamp)
        if not is_miss:
            self.learning_rate_controller.hit_rate += 1
        if is_miss:
            self.pollution_handler.increment_unique_count()
        self.pollution_handler.set_unique(data_block)
        if self.time % self.cache_size == 0:
            self.pollution_handler.update(self.time)
        operation = CacheHandler.INSERT if is_miss else CacheHandler.HIT
        return operation, evicted

    def get_queue_size(self):
        x, y = zip(*self.visualizer.get('queue_size'))
        return y

