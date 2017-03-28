from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
from contextlib import contextmanager
import pickle
from functools import partial, wraps


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


def maybe_load(save_path):
    def decorator(f):
        @wraps(f)
        def wrapper(*args, **kwargs):
            if _check_file(save_path):
                obj = load_pickle(save_path)
                return obj
            else:
                obj = f(*args, **kwargs)
                save_pickle(obj, save_path)
        return wrapper
    return decorator
