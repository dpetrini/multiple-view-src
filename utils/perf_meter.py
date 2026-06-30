#
#   Instrumentation (performance measurement) functions. Decorators to measure average delays.
#

import time
import numpy as np


def clock(func):
    """ Decorator to measure time """
    def clocked(*args):
        t_0 = time.perf_counter()
        result = func(*args)
        print('Elapsed time [{}]: {:.4f}s'.format(func.__name__, time.perf_counter()-t_0))
        return result
    return clocked

def avg_ten_ms(func):
    """ Decorator to measure average time considering 10 calls
        in miliseconds (1.2e)
    """
    callings = []
    def clocked(*args):
        t_0 = time.time()
        result = func(*args)
        callings.append(time.time()-t_0)
        if not len(callings) % 10:
            mean_ten = np.mean(np.array(callings[-10:]))*1000
            print('Elapsed time [10x{}]: {:.3f} ms'.format(func.__name__, mean_ten))
            callings.clear()
        return result
    return clocked

# global_timer = []
def avg_100_ms(func):
    """ Decorator to measure average time considering 100 calls
        in miliseconds (1.2e)
    """
    callings = []
    def clocked(*args):
        t_0 = time.time()
        result = func(*args)
        callings.append(time.time()-t_0)
        # global_timer.append(time.perf_counter()-t_0)
        if not len(callings) % 100:
            mean_100 = np.mean(np.array(callings[-100:]))*1000
            # print(f'Avg per frame [100x{func.__name__}]: {mean_100:.3f} ms')
            print('Avg per frame [100x{}]: {:.3f} ms'.format(func.__name__, mean_100))
            # callings.clear()
            del callings[:]
        return result
    return clocked