from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import time
import pickle
import threading
from contextlib import contextmanager
from functools import wraps
from collections import deque

import tensorflow as tf


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


def start_threads(thread_fn, args, n_threads=1):
    assert n_threads == 1, "Having multiple threads causes duplicate data in the queue."

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


def set_logging_verbosity(logging_verbosity="INFO"):
    if logging_verbosity == "INFO":
        tf.logging.set_verbosity(tf.logging.INFO)
    elif logging_verbosity == "WARN":
        tf.logging.set_verbosity(tf.logging.WARN)
    elif logging_verbosity == "ERROR":
        tf.logging.set_verbosity(tf.logging.ERROR)
    elif logging_verbosity == "DEBUG":
        tf.logging.set_verbosity(tf.logging.DEBUG)


class MovingAverage(object):

    def __init__(self, size):
        """
        Initialize your data structure here.
        :type size: int
        """
        self.__size = size
        self.__sum = 0
        self.__q = deque([])

    def next(self, val):
        """
        :type val: int
        :rtype: float
        """
        if len(self) == self.__size:
            self.__sum -= self.__q.popleft()
        self.__sum += val
        self.__q.append(val)
        return 1.0 * self.__sum / len(self.__q)

    def __len__(self):
        return len(self.__q)


def count_number_of_parameters():
    total_parameters = 0
    for variable in tf.trainable_variables():
        # shape is an array of tf.Dimension
        shape = variable.get_shape()
        variable_parametes = 1
        for dim in shape:
            variable_parametes *= dim.value
        total_parameters += variable_parametes
    return total_parameters
    

def delete_files():
    pass