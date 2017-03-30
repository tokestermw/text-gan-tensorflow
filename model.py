""" Text GAN

Adverserial networks applied to language models using Gumbel Softmax.

Can be used as pure language model.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from collections import namedtuple

import tensorflow as tf

# -- local imports
from data_loader import build_vocab, preprocess, get_and_run_input_queues
from layers import (
    embedding_layer, recurrent_layer, reshape_layer, dense_layer, softmax_layer,
    cross_entropy_layer, sigmoid_cross_entropy_layer, mean_loss_by_example_layer, dropout_layer, relu_layer,
)

# TODO: add to flags
HIDDEN_DIMS = 128

GeneratorTuple = namedtuple("Generator", 
    ["rnn_outputs", "flat_logits", "probs", "loss"])
DiscriminatorTuple = namedtuple("Discriminator", 
    ["rnn_final_state", "prediction_logits", "loss"])


class Model:
    def __init__(self, path, **model_params):
        self.path = path
        self.model_params = model_params

        self.word2idx, self.idx2word = build_vocab(self.path)
        self.vocab_size = len(self.word2idx)

        self.enqueue_data, self.source, self.target, self.sequence_length = \
            prepare_data(self.path, self.word2idx)

        self.generator_template = tf.make_template("generator", generator)
        self.discriminator_template = tf.make_template("discriminator", discriminator)

        self.g_tensors = self.generator_template(self.source, self.target, self.sequence_length, self.vocab_size, is_pretrain=True)
        self.d_tensors = self.discriminator_template(self.g_tensors[0], self.sequence_length, self.vocab_size, is_real=True)


def prepare_data(path, word2idx, batch_size=32):
    with tf.device("/cpu:0"):
        enqueue_data, dequeue_batch = get_and_run_input_queues(path, word2idx, batch_size=batch_size)
        source, target, sequence_length = preprocess(dequeue_batch)
    return enqueue_data, source, target, sequence_length


def prepare_decoder():
    from decoders import gumbel_output_fn

    pass


# TODO: generalize model_params
def generator(source, target, sequence_length, vocab_size, is_pretrain=True):
    rnn_outputs = (
        source >>
        embedding_layer(vocab_size, HIDDEN_DIMS, name="embedding_matrix") >>
        recurrent_layer(sequence_length=sequence_length)
    )

    flat_logits = (
        rnn_outputs >>
        reshape_layer(shape=(-1, HIDDEN_DIMS)) >>
        dense_layer(hidden_dims=vocab_size, name="output_projections")
    )

    probs = flat_logits >> softmax_layer() >> reshape_layer(shape=tf.shape(target))

    loss = (
        flat_logits >> 
        cross_entropy_layer(target=target) >>
        reshape_layer(shape=tf.shape(target)) >>
        mean_loss_by_example_layer(sequence_length=sequence_length)
    )

    return GeneratorTuple(rnn_outputs=rnn_outputs, flat_logits=flat_logits, probs=probs, loss=loss)


def discriminator(input_vectors, sequence_length, vocab_size, is_real=True):
    """
    Args:
    """
    rnn_final_state = (
        input_vectors >> 
        dense_layer(hidden_dims=HIDDEN_DIMS) >>  # projection layer keep shape [B, T, H]
        recurrent_layer(sequence_length=sequence_length, return_final_state=True)
    )

    prediction_logits = (
        rnn_final_state >>
        dense_layer(hidden_dims=HIDDEN_DIMS) >> 
        relu_layer() >>
        dropout_layer() >>
        dense_layer(hidden_dims=HIDDEN_DIMS) >>
        relu_layer() >>
        dropout_layer() >>
        dense_layer(hidden_dims=1)
    )

    target = tf.zeros(shape=tf.shape(prediction_logits), dtype=tf.float32)
    target += 1.0 if is_real else 0.0

    loss = (
        prediction_logits >>
        sigmoid_cross_entropy_layer(target=target)
    )

    return DiscriminatorTuple(rnn_final_state=rnn_final_state, prediction_logits=prediction_logits, loss=loss)


if __name__ == "__main__":
    from data_loader import DATA_PATH
    path = DATA_PATH["ptb"]["train"]

    model = Model(path)