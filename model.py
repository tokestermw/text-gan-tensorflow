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
from tensorflow.contrib.framework.python.ops.variables import get_or_create_global_step

# -- local imports
from data_loader import get_corpus_size, build_vocab, preprocess, get_input_queues
import layers as lay
from decoders import gumbel_decoder_fn

GENERATOR_PREFIX = "generator"
DISCRIMINATOR_PREFIX = "discriminator"

GeneratorTuple = namedtuple("Generator",
                            ["rnn_outputs", "flat_logits", "probs", "loss", "embedding_matrix", "output_projections"])
DiscriminatorTuple = namedtuple("Discriminator",
                                ["rnn_final_state", "prediction_logits", "loss"])


# TODO: separate the variables for generator and discriminators
class Model:
    def __init__(self, corpus, **opts):
        self.corpus = corpus

        self.opts = opts

        self.global_step = get_or_create_global_step()
        self.increment_global_step_op = tf.assign(self.global_step, self.global_step + 1, name="increment_global_step")

        self.corpus_size = get_corpus_size(self.corpus["train"])
        self.corpus_size_valid = get_corpus_size(self.corpus["valid"])

        self.word2idx, self.idx2word = build_vocab(self.corpus["train"])
        self.vocab_size = len(self.word2idx)

        self.generator_template = tf.make_template(GENERATOR_PREFIX, generator)
        self.discriminator_template = tf.make_template(DISCRIMINATOR_PREFIX, discriminator)

        self.enqueue_data, _, source, target, sequence_length = \
            prepare_data(self.corpus["train"], self.word2idx, num_threads=7, **self.opts)

        # TODO: option to either do pretrain or just generate?
        self.g_tensors_pretrain = self.generator_template(
            source, target, sequence_length, self.vocab_size, **self.opts)

        self.enqueue_data_valid, self.input_ph, source_valid, target_valid, sequence_length_valid = \
            prepare_data(self.corpus["valid"], self.word2idx, num_threads=1, **self.opts)

        self.g_tensors_pretrain_valid = self.generator_template(
            source_valid, target_valid, sequence_length_valid, self.vocab_size, **self.opts)

        self.decoder_fn = prepare_custom_decoder(
            sequence_length, self.g_tensors_pretrain.embedding_matrix, self.g_tensors_pretrain.output_projections)

        self.g_tensors_fake = self.generator_template(
            source, target, sequence_length, self.vocab_size, decoder_fn=self.decoder_fn, **self.opts)

        # TODO: using the rnn outputs from pretraining as "real" instead of target embeddings (aka professor forcing)
        self.d_tensors_real = self.discriminator_template(
            self.g_tensors_pretrain.rnn_outputs, sequence_length, is_real=True, **self.opts)

        # TODO: check to see if sequence_length is correct
        self.d_tensors_fake = self.discriminator_template(
            self.g_tensors_fake.rnn_outputs, None, is_real=False, **self.opts)

        self.g_tvars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=GENERATOR_PREFIX)
        self.d_tvars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=DISCRIMINATOR_PREFIX)


def prepare_data(path, word2idx, num_threads=8, **opts):
    with tf.device("/cpu:0"):
        enqueue_data, dequeue_batch = get_input_queues(
            path, word2idx, batch_size=opts["batch_size"], num_threads=num_threads)
        # TODO: put this logic somewhere else
        input_ph = tf.placeholder_with_default(dequeue_batch, (None, None))
        source, target, sequence_length = preprocess(input_ph)
    return enqueue_data, input_ph, source, target, sequence_length


def prepare_custom_decoder(sequence_length, embedding_matrix, output_projections):
    # TODO: this is brittle, global variables
    cell = tf.get_collection("rnn_cell")[0]
    encoder_state = cell.zero_state(tf.shape(sequence_length)[0], tf.float32)

    # embedding_matrix = tf.get_collection("embedding_matrix")[0]
    # output_projections = tf.get_collection("output_projections")[:2]  # TODO: repeated output_projections

    maximum_length = tf.reduce_max(sequence_length) + 3

    decoder_fn = gumbel_decoder_fn(encoder_state, embedding_matrix, output_projections, maximum_length)
    return decoder_fn


def generator(source, target, sequence_length, vocab_size, decoder_fn=None, **opts):
    """
    Args:
        source: TensorFlow queue or placeholder tensor for word ids for source 
        target: TensorFlow queue or placeholder tensor for word ids for target
        sequence_length: TensorFlow queue or placeholder tensor for number of word ids for each sentence
        vocab_size: max vocab size determined from data
        decoder_fn: if using custom decoder_fn else use the default dynamic_rnn
    """
    tf.logging.info(" Setting up generator")

    embedding_layer = lay.embedding_layer(vocab_size, opts["embedding_dim"], name="embedding_matrix")

    # TODO: add batch norm?
    rnn_outputs = (
        source >>
        embedding_layer >>
        lay.word_dropout_layer(keep_prob=opts["word_dropout_keep_prob"]) >>
        lay.recurrent_layer(hidden_dims=opts["rnn_hidden_dim"], keep_prob=opts["recurrent_dropout_keep_prob"],
                            sequence_length=sequence_length, decoder_fn=decoder_fn, name="rnn_cell")
    )

    output_projection_layer = lay.dense_layer(hidden_dims=vocab_size, name="output_projections")

    flat_logits = (
        rnn_outputs >>
        lay.reshape_layer(shape=(-1, opts["rnn_hidden_dim"])) >>
        output_projection_layer
    )

    probs = flat_logits >> lay.softmax_layer()

    embedding_matrix = embedding_layer.get_variables_in_scope()
    output_projections = output_projection_layer.get_variables_in_scope()

    if decoder_fn is not None:
        return GeneratorTuple(rnn_outputs=rnn_outputs, flat_logits=flat_logits, probs=probs, loss=None,
                              embedding_matrix=embedding_matrix[0], output_projections=output_projections)

    loss = (
        flat_logits >>
        lay.cross_entropy_layer(target=target) >>
        lay.reshape_layer(shape=tf.shape(target)) >>
        lay.mean_loss_by_example_layer(sequence_length=sequence_length)
    )

    # TODO: add dropout penalty
    return GeneratorTuple(rnn_outputs=rnn_outputs, flat_logits=flat_logits, probs=probs, loss=loss,
                          embedding_matrix=embedding_matrix[0], output_projections=output_projections)


def discriminator(input_vectors, sequence_length, is_real=True, **opts):
    """
    Args:
        input_vectors: output of the RNN either from real or generated data
        sequence_length: TensorFlow queue or placeholder tensor for number of word ids for each sentence
        is_real: if True, RNN outputs when feeding in actual data, if False feeds in generated data
    """
    tf.logging.info(" Setting up discriminator")

    rnn_final_state = (
        input_vectors >>
        lay.dense_layer(hidden_dims=opts["embedding_dim"]) >>
        lay.recurrent_layer(sequence_length=sequence_length, hidden_dims=opts["rnn_hidden_dim"],
                            return_final_state=True)
    )

    prediction_logits = (
        rnn_final_state >>
        lay.dense_layer(hidden_dims=opts["output_hidden_dim"]) >>
        lay.relu_layer() >>
        lay.dropout_layer(opts["output_dropout_keep_prob"]) >>
        lay.dense_layer(hidden_dims=opts["output_hidden_dim"]) >>
        lay.relu_layer() >>
        lay.dropout_layer(opts["output_dropout_keep_prob"]) >>
        lay.dense_layer(hidden_dims=1)
    )

    if is_real:
        target = tf.ones_like(prediction_logits)
    else:
        target = tf.zeros_like(prediction_logits)

    # TODO: add accuracy
    loss = (
        prediction_logits >>
        lay.sigmoid_cross_entropy_layer(target=target)
    )

    # TODO: return logits in case for WGAN and l2 GANs
    return DiscriminatorTuple(rnn_final_state=rnn_final_state, prediction_logits=prediction_logits, loss=loss)


if __name__ == "__main__":
    from data_loader import DATA_PATH
    from train import opts

    corpus = DATA_PATH["ptb"]

    model = Model(corpus, **opts)
