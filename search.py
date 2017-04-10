""" Train
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf

flags = tf.flags

# -- saver options
# flags.DEFINE_string("model_dir", "./tmp", (
    # "Model directory."))

# FLAGS = flags.FLAGS
# opts = FLAGS.__flags  # dict TODO: make class?

# set_logging_verbosity(FLAGS.logging_verbosity)

MAX_GEN = 40
_END_ID = 3


def reverse_decode(vector, rev_vocab):
    tokens = [rev_vocab[i] for i in vector.tolist()]
    return " ".join(tokens)


# TODO: only works for one example
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


if __name__ == "__main__":
    from data_loader import DATA_PATH
    from train import opts
    corpus = DATA_PATH["ptb"]

    model = Model(corpus, **opts)

