""" Text GAN

Adverserial networks applied to language models using Gumbel Softmax.

Can be used as pure language model.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import functools
from collections import namedtuple

import tensorflow as tf

# -- local imports
from data_loader import build_vocab, preprocess, get_and_run_input_queues
from layers import (
    embedding_layer, recurrent_layer, reshape_layer, dense_layer, softmax_layer,
    cross_entropy_layer, sigmoid_cross_entropy_layer, mean_loss_by_example_layer, 
    dropout_layer, word_dropout_layer, relu_layer, tanh_layer
)
from decoders import gumbel_decoder_fn

GeneratorTuple = namedtuple("Generator", 
    ["rnn_outputs", "flat_logits", "probs", "loss"])
DiscriminatorTuple = namedtuple("Discriminator", 
    ["rnn_final_state", "prediction_logits", "loss"])


class Model:
    def __init__(self, path, **opts):
        self.path = path
        self.opts = opts

        self.word2idx, self.idx2word = build_vocab(self.path)
        self.vocab_size = len(self.word2idx)

        self.enqueue_data, self.source, self.target, self.sequence_length = \
            prepare_data(self.path, self.word2idx, **self.opts)

        self.generator_template = tf.make_template("generator", generator)
        self.discriminator_template = tf.make_template("discriminator", discriminator)

        self.g_tensors_pretrain = self.generator_template(
            self.source, self.target, self.sequence_length, self.vocab_size, **self.opts)

        self.decoder_fn = prepare_custom_decoder(self.sequence_length)

        self.g_tensors_generated = self.generator_template(
            self.source, self.target, self.sequence_length, self.vocab_size, decoder_fn=self.decoder_fn, **self.opts)

        # TODO: using the rnn outputs from pretraining as "real" instead of target embeddings (aka professor forcing)
        self.d_tensors_real = self.discriminator_template(
            self.g_tensors_pretrain.rnn_outputs, self.sequence_length, is_real=True, **self.opts)

        # TODO: check to see if sequence_length is correct
        self.d_tensors_generated = self.discriminator_template(
            self.g_tensors_generated.rnn_outputs, None, is_real=False, **self.opts)


def prepare_data(path, word2idx, **opts):
    with tf.device("/cpu:0"):
        enqueue_data, dequeue_batch = get_and_run_input_queues(
            path, word2idx, batch_size=opts["batch_size"], epoch_size=opts["epoch_size"])
        source, target, sequence_length = preprocess(dequeue_batch)
    return enqueue_data, source, target, sequence_length


def prepare_custom_decoder(sequence_length):
    # TODO: is this confusing? global variables
    cell = tf.get_collection("rnn_cell")[0]
    encoder_state = cell.zero_state(tf.shape(sequence_length)[0], tf.float32)

    embedding_matrix = tf.get_collection("embedding_matrix")[0]
    output_projections = tf.get_collection("output_projections")

    maximum_length = tf.reduce_max(sequence_length) + 3

    decoder_fn = gumbel_decoder_fn(encoder_state, embedding_matrix, output_projections, maximum_length)
    return decoder_fn


def prepare_inputs():
    pass


def generator(source, target, sequence_length, vocab_size, decoder_fn=None, **opts):
    """
    Args:
        source: TensorFlow queue or placeholder tensor for word ids for source 
        target: TensorFlow queue or placeholder tensor for word ids for target
        sequence_length: TensorFlow queue or placeholder tensor for word ids for target
        vocab_size: max vocab size determined from data
        decoder_fn: if using custom decoder_fn else use the default dynamic_rnn
    """
    tf.logging.info(" --- Setting up generator")

    rnn_outputs = (
        source >>
        embedding_layer(vocab_size, opts["embedding_dim"], name="embedding_matrix") >>
        word_dropout_layer(keep_prob=opts["word_dropout_keep_prob"]) >>
        recurrent_layer(hidden_dims=opts["rnn_hidden_dim"], keep_prob=opts["recurrent_dropout_keep_prob"], 
            sequence_length=sequence_length, decoder_fn=decoder_fn, name="rnn_cell")
    )

    flat_logits = (
        rnn_outputs >>
        reshape_layer(shape=(-1, opts["rnn_hidden_dim"])) >>
        dense_layer(hidden_dims=vocab_size, name="output_projections")
    )

    probs = flat_logits >> softmax_layer() >> reshape_layer(shape=tf.shape(target))

    if decoder_fn is not None:
        return GeneratorTuple(rnn_outputs=rnn_outputs, flat_logits=flat_logits, probs=probs, loss=None)

    # TODO: return cost
    loss = (
        flat_logits >> 
        cross_entropy_layer(target=target) >>
        reshape_layer(shape=tf.shape(target)) >>
        mean_loss_by_example_layer(sequence_length=sequence_length)
    )

    return GeneratorTuple(rnn_outputs=rnn_outputs, flat_logits=flat_logits, probs=probs, loss=loss)


def discriminator(input_vectors, sequence_length, is_real=True, **opts):
    """
    Args:
        input_vectors:
        sequence_length:
        is_real: 
    """
    tf.logging.info(" --- Setting up discriminator")

    rnn_final_state = (
        input_vectors >> 
        dense_layer(hidden_dims=opts["embedding_dim"]) >>  # projection layer keep shape [B, T, H]
        recurrent_layer(sequence_length=sequence_length, hidden_dims=opts["rnn_hidden_dim"], return_final_state=True)
    )

    prediction_logits = (
        rnn_final_state >>
        dense_layer(hidden_dims=opts["output_hidden_dim"]) >> 
        relu_layer() >>
        dropout_layer(opts["output_dropout_keep_prob"]) >>
        dense_layer(hidden_dims=opts["output_hidden_dim"]) >>
        relu_layer() >>
        dropout_layer(opts["output_dropout_keep_prob"]) >>
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
    from train import opts
    path = DATA_PATH["ptb"]["train"]

    model = Model(path, **opts)