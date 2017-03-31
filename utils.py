from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import time
import pickle
import threading
from contextlib import contextmanager
from functools import wraps


# TODO: move to utils
def _check_file(path):
    return os.path.isfile(path)


def save_pickle(obj, path):
    with open(path, 'wb') as f:
        pickle.dump(obj, f)


def load_pickle(path):
    with open(path, 'rb') as f:
        obj = pickle.load(f)
    return obj


def maybe_save(save_path):
    def decorator(f):
        @wraps(f)
        def wrapper(*args, **kwargs):
            if _check_file(save_path):
                obj = load_pickle(save_path)
            else:
                obj = f(*args, **kwargs)
                save_pickle(obj, save_path)
            return obj
        return wrapper
    return decorator


def start_threads(thread_fn, args, n_threads=8):
    threads = []
    for n in range(n_threads):
        t = threading.Thread(target=thread_fn, args=args)
        t.daemon = True  # thread will close when parent quits
        t.start()
        threads.append(t)

    time.sleep(1)  # enqueue a bunch before dequeue
    return threads


def compose(data, *funcs):
    for func in funcs:
        data = func(data)
    return data