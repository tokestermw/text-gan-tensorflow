""" TensorFlow Layers

Convenience functions but Input and Output should be tensors.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from contextlib import contextmanager
from functools import partial, wraps

import tensorflow as tf
from tensorflow.contrib import seq2seq


_phase = tf.Variable(False, name='phase', trainable=False, collections=[tf.GraphKeys.LOCAL_VARIABLES])
_phase_train = _phase.assign(True)
_phase_infer = _phase.assign(False)


# TODO: move to ops
def _rank(x):
    return len(x.get_shape())


def _apply_dropout_mask(tensor_shape, keep_prob=1.0):
    random_tensor = keep_prob + tf.random_uniform(tensor_shape, dtype=tf.float32)
    binary_mask = tf.floor(random_tensor)
    binary_mask = tf.reciprocal(keep_prob) * binary_mask
    return binary_mask


def _global_keep_prob(keep_prob):
    keep_prob = tf.convert_to_tensor(keep_prob, dtype=tf.float32)
    keep_prob = tf.cond(_phase, lambda: keep_prob, lambda: keep_prob * 0.0 + 1.0)
    return keep_prob


def pipe(func):

    class Pipe(object):
        def __init__(self, *args, **kwargs):
            self.func = func
            self.args = args
            self.kwargs = kwargs
            self.name = self.kwargs.get("name", self.func.__name__)

        def __rrshift__(self, other):
            # >>
            with tf.variable_scope(self.name, None, self.args):
                out = self.func(other, *self.args, **self.kwargs)
            return out

    return Pipe


@pipe
def identity_layer(tensor, **opts):
    out = tf.identity(tensor)
    return out


@pipe
def embedding_layer(tensor, vocab_size=None, embedding_dim=None, embedding_matrix=None, **opts):
    if embedding_matrix is None:
        initializer = tf.contrib.layers.xavier_initializer(uniform=True)
        embedding_matrix = initializer(shape=(vocab_size, embedding_dim))

    if opts.get("name"):
        tf.add_to_collection(opts.get("name"), embedding_matrix)

    out = tf.nn.embedding_lookup(embedding_matrix, tensor)
    return out 


@pipe
def recurrent_layer(tensor, cell=None, hidden_dims=128, sequence_length=None, decoder_fn=None, 
                    activation=tf.nn.tanh, initializer=tf.orthogonal_initializer(), initial_state=None, 
                    keep_prob=1.0,
                    return_final_state=False, return_next_cell_input=True, **opts):
    if cell is None:
        cell = tf.contrib.rnn.BasicRNNCell(hidden_dims, activation=activation)

    if keep_prob < 1.0:
        keep_prob = _global_keep_prob(keep_prob)
        cell = tf.contrib.rnn.DropoutWrapper(cell, keep_prob, keep_prob)

    if opts.get("name"):
        tf.add_to_collection(opts.get("name"), cell)

    if decoder_fn is None:
        outputs, final_state = tf.nn.dynamic_rnn(cell, tensor, 
            sequence_length=sequence_length, initial_state=initial_state, dtype=tf.float32)
        final_context_state = None
    else:
        # TODO: turn off sequence_length?
        outputs, final_state, final_context_state = seq2seq.dynamic_rnn_decoder(
            cell, decoder_fn, inputs=None, sequence_length=sequence_length)

    if return_final_state:
        return final_state
    else:
        return outputs


@pipe
def reshape_layer(tensor, shape, **opts):
    out = tf.reshape(tensor, shape=shape)
    return out


@pipe
def dense_layer(tensor, hidden_dims, weight=None, bias=None, **opts):
    original_tensor_shape = tf.shape(tensor)
    in_dim = int(tensor.get_shape()[-1])

    rank = _rank(tensor)
    if rank > 2:
        # -- time distributed dense
        tensor = tf.reshape(tensor, shape=(-1, in_dim))

    if weight is None:
        initializer = tf.contrib.layers.xavier_initializer(uniform=True)
        weight = initializer(shape=(in_dim, hidden_dims))
    if bias is None:
        bias = tf.zeros(shape=hidden_dims)

    if opts.get("name"):
        tf.add_to_collection(opts.get("name"), weight)
        tf.add_to_collection(opts.get("name"), bias)

    out = tf.add(tf.matmul(tensor, weight), bias)

    if rank > 2:
        # reshape back to time dimension
        out = tf.reshape(out, shape=original_tensor_shape)

    return out


@pipe
def dropout_layer(tensor, keep_prob=1.0, **opts):
    keep_prob = _global_keep_prob(keep_prob)
    out = tf.nn.dropout(tensor, keep_prob=keep_prob)
    return out


# TODO: should i normalize?
@pipe
def word_dropout_layer(tensor, keep_prob=1.0, **opts):
    keep_prob = _global_keep_prob(keep_prob)

    rank = _rank(tensor)
    assert rank == 3, "Use embedding lookup layer"

    binary_mask = _apply_dropout_mask(tf.shape(tensor)[:2], keep_prob)
    binary_mask = tf.expand_dims(binary_mask, axis=-1)  # proper broadcasting to zero out entire word vectors

    out = tensor * binary_mask
    return out


@pipe
def relu_layer(tensor):
    out = tf.nn.relu(tensor)
    return out


@pipe
def tanh_layer(tensor):
    out = tf.nn.tanh(tensor)
    return out


@pipe
def softmax_layer(tensor, softmax_func=None, **opts):
    if softmax_func is None:
        softmax_func = tf.nn.softmax

    out = softmax_func(tensor)
    return out


@pipe
def cross_entropy_layer(tensor, target, **opts):
    if _rank(tensor) > 1:
        target = tf.reshape(target, shape=(-1, ))

    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=tensor, labels=target)
    mask = tf.cast(tf.not_equal(target, tf.zeros_like(target)), dtype=tf.float32)
    out = cross_entropy * mask
    return out


@pipe
def sigmoid_cross_entropy_layer(tensor, target, **opts):
    out = tf.nn.sigmoid_cross_entropy_with_logits(logits=tensor, labels=target)
    return out


@pipe
def mean_loss_by_example_layer(tensor, sequence_length, **opts):
    out = tf.div(
        tf.reduce_sum(tensor, axis=1),
        tf.cast(sequence_length, dtype=tf.float32)
    )
    return out


@pipe
def conv1d_layer(tensor, dilation_rate=1, **opts):
    raise NotImplementedError


@pipe
def residual_layer(tensor, **opts):
    raise NotImplementedError


@pipe
def highway_layer(tensor, **opts):
    raise NotImplementedError


if __name__ == "__main__":
    import numpy as np

    batch_size = 10
    sequence_length = 5
    vocab_size = 100
    embedding_dim = 32

    word_ids = np.random.randint(0, vocab_size, batch_size * sequence_length).reshape(batch_size, sequence_length)
    tensor = tf.constant(word_ids)

    # print(word_ids >> identity_layer() >> embedding_layer(vocab_size, embedding_dim))
    print(tensor >> identity_layer() >> embedding_layer(vocab_size, embedding_dim))
