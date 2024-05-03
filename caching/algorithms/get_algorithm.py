from .min import MIN
from .lru import LRU
from .lfu import LFU
from .mru import MRU
from .lecar import LeCaR
from .lirs import LIRS
from .ml_algo import ml_algo


def get_algorithm(alg_name):
    alg_name = alg_name.lower()

    if alg_name == 'min':
        return MIN
    if alg_name == 'lru':
        return LRU
    if alg_name == 'lfu':
        return LFU
    if alg_name == 'mru':
        return MRU
    if alg_name == 'lecar':
        return LeCaR
    if alg_name == 'lirs':
        return LIRS
    if alg_name == 'ml_algo':
        return ml_algo
    return None
