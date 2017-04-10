""" Train
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf

MAX_GEN = 40
_END_ID = 3


def reverse_decode(vector, rev_vocab):
    tokens = [rev_vocab[i] for i in vector.tolist()]
    return " ".join(tokens)


# TODO: only works for one example
# TODO: tensorflow native op
def greedy_argmax(vector, step):
    counter = 0
    while counter < MAX_GEN:
        probs = step(vector)
        last_vector = probs[-1, :]
        last_id = np.argmax(last_vector)
        vector = np.concatenate([vector, [last_id]])
        if last_id == _END_ID:
            break
        counter += 1
    return vector


def beam_search():
    pass
