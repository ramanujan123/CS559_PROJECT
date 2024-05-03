from algorithms.lib.traces import identify_trace, get_trace_reader
from caching.algorithms.lib.cachehandler import CacheOp
from algorithms.get_algorithm import get_algorithm
from itertools import product
import numpy as np
import csv
import os


class AlgorithmTester:
    def __init__(self, algorithm, cache_size, trace_file, output_csv):
        self.algorithm = algorithm
        self.cache_size = cache_size
        self.trace_file = trace_file

        trace_type = identify_trace(trace_file)
        trace_reader = get_trace_reader(trace_type)
        self.reader = trace_reader(trace_file)
        self.output_csv = output_csv

        self.misses = 0

    def _run(self):
        alg = get_algorithm(self.algorithm)(self.cache_size)
        for lba, write in self.reader.read():
            op, evicted = alg.request(lba)
            if op != CacheOp.HIT:
                self.misses += 1

    def run_test(self):
        self._run()
        ios = self.reader.num_requests()
        hits = ios - self.misses
        print(
            "Results: {:<10} size={:<8} hits={}, misses={}, hitrate={:4}% {}".
            format(self.algorithm, self.cache_size, hits, self.misses,
                   round(100 * hits / ios, 2), self.trace_file))

        if self.output_csv:
            self.write_csv(self.output_csv, hits, ios)

    def write_csv(self, filename, hits, ios):
        with open(filename, 'a+') as csvfile:
            writer = csv.writer(csvfile,
                                delimiter=',',
                                quotechar='|',
                                quoting=csv.QUOTE_MINIMAL)
            writer.writerow([
                self.trace_file, self.algorithm, hits, self.misses,
                self.cache_size,
                round(100 * hits / ios, 2)
            ])


def run_entire_trace(trace_name):
    trace_type = identify_trace(trace_name)
    trace_reader = get_trace_reader(trace_type)
    reader = trace_reader(trace_name)

    for lba, write in reader.read():
        pass
    return reader


def get_unique_count(trace_name):
    reader = run_entire_trace(trace_name)
    return reader.num_unique()


def get_reuse_count(trace_name):
    reader = run_entire_trace(trace_name)
    return reader.num_reuse()


def generate_trace_names(trace):
    if trace.startswith('~'):
        trace = os.path.expanduser(trace)

    if os.path.isdir(trace):
        for trace_name in os.listdir(trace):
            yield os.path.join(trace, trace_name)
    elif os.path.isfile(trace):
        yield trace
    else:
        raise ValueError("{} is not a directory or a file".format(trace))


def generate_algorithm_tests(algorithm, cache_size, trace_name, output_csv):
    yield AlgorithmTester(algorithm, cache_size, trace_name, output_csv)


if __name__ == '__main__':
    import sys
    import json
    import math
    import os

    output_csv = None

    algorithm = sys.argv[1]

    cache_size_str = sys.argv[2]
    try:
        cache_size = int(cache_size_str)
    except:
        cache_size = float(cache_size_str)

    request_count_type = sys.argv[3]

    trace = sys.argv[4]

    if len(sys.argv) > 5:
        output_csv = sys.argv[5]

    # TODO revisit and cleanup
    if request_count_type == 'reuse':
        request_counter = get_reuse_count
    elif request_count_type == 'unique':
        request_counter = get_unique_count
    elif request_count_type == 'size':
        request_counter = None
    else:
        raise ValueError(
            "Unknown request_count_type found: {}".format(request_count_type))

    for trace_name in generate_trace_names(trace):
        if isinstance(cache_size, float):
            count = request_counter(trace_name)

        tmp_cache_size = cache_size
        if isinstance(cache_size, float):
            cache_size = math.floor(cache_size * count)
        if cache_size < 4:
            print(
                "Cache size {} too small for trace {}. Calculated size is {}. Skipping"
                .format(tmp_cache_size, trace_name, cache_size),
                file=sys.stderr)
            continue

        for test in generate_algorithm_tests(algorithm, cache_size, trace_name,
                                           output_csv):
            test.run_test()

