""" TensorFlow Layers

Convenience functions but Input and Output should be tensors.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from contextlib import contextmanager
from functools import partial

import tensorflow as tf


def pipe(func):
    func = func

    class Pipe(object):
        def __init__(self, *args, **kwargs):
            self.func = func
            self.args = args
            self.kwargs = kwargs

        def __rrshift__(self, other):
            return self.func(other, *self.args, **self.kwargs)

        def __ror__(self, other):
            # doesn't work for TensorFlow or NumPy object
            return self.func(other, *self.args, **self.kwargs)

    return Pipe


@pipe
def identity_layer(tensor):
    return tensor


@pipe
def embedding_layer(tensor, vocab_size, embedding_dim):
    initializer = tf.contrib.layers.xavier_initializer(uniform=True)
    embedding_matrix = initializer(shape=(vocab_size, embedding_dim))
    out = tf.nn.embedding_lookup(embedding_matrix, tensor)
    return out 


if __name__ == "__main__":
    import numpy as np

    batch_size = 10
    sequence_length = 5
    vocab_size = 100
    embedding_dim = 32

    word_ids = np.random.randint(0, vocab_size, batch_size * sequence_length).reshape(batch_size, sequence_length)
    tensor = tf.constant(word_ids)

    print(word_ids >> identity_layer() >> embedding_layer(vocab_size, embedding_dim))
    print(tensor >> identity_layer() >> embedding_layer(vocab_size, embedding_dim))
