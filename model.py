""" Text GAN

Adverserial networks applied to language models using Gumbel Softmax.

Can be used as pure language model.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

# -- local imports
from data_loader import build_vocab, preprocess, get_and_run_input_queues
from layers import (
    embedding_layer, recurrent_layer, reshape_layer, dense_layer, softmax_layer,
    cross_entropy_layer, mean_loss_by_example_layer
)

# TODO: add to flags
HIDDEN_DIMS = 128


class Model:
    def __init__(self, path, **model_params):
        self.path = path
        self.model_params = model_params

        self.word2idx, self.idx2word = build_vocab(self.path)
        self.vocab_size = len(self.word2idx)

        self.enqueue_data, self.source, self.target, self.sequence_length = \
            prepare_data(self.path, self.word2idx)

        with tf.variable_scope("generator"):
            # TODO: rename
            g_tensors = generator(self.source, self.target, self.sequence_length, self.vocab_size)
            self.flat_logits, self.probs, self.loss, self.cost = g_tensors

        optim = tf.train.AdamOptimizer(learning_rate=0.005)
        self.train_op = optim.minimize(self.cost)

        d_tensors = discriminator()


def prepare_data(path, word2idx, batch_size=32):
    with tf.device("/cpu:0"):
        enqueue_data, dequeue_batch = get_and_run_input_queues(path, word2idx, batch_size=batch_size)
        source, target, sequence_length = preprocess(dequeue_batch)
    return enqueue_data, source, target, sequence_length


def generator(source, target, sequence_length, vocab_size):
    flat_logits = (
        source >>
        embedding_layer(vocab_size, HIDDEN_DIMS) >>
        recurrent_layer(sequence_length=sequence_length) >> 
        reshape_layer(shape=(-1, HIDDEN_DIMS)) >>
        dense_layer(hidden_dims=vocab_size)
    )

    probs = flat_logits >> softmax_layer() >> reshape_layer(shape=tf.shape(target))

    loss = (
        flat_logits >> 
        cross_entropy_layer(target=target) >>
        reshape_layer(shape=tf.shape(target)) >>
        mean_loss_by_example_layer(sequence_length=sequence_length)
    )

    cost = tf.reduce_mean(loss)
    return flat_logits, probs, loss, cost


def discriminator():
    pass


if __name__ == "__main__":
    from data_loader import DATA_PATH
    path = DATA_PATH["ptb"]["train"]

    model = Model(path)