""" RNN decoder function using Gumbel / Concrete distribution outputs.

So the output of the generation should be random.

Can be used for inputs for GAN.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import functools
from collections import namedtuple

import tensorflow as tf
from tensorflow.python.framework import ops, dtypes
from tensorflow.python.util import nest
from tensorflow.python.ops import array_ops, control_flow_ops, variable_scope as vs, math_ops

# -- local import
from distributions import gumbel_softmax


# def gumbel_output_fn(cell_output, embedding_matrix, output_projections, 
                     # end_of_sequence_id=3, temperature=1.0):
def gumbel_decoder_fn(encoder_state, embedding_matrix, output_projections, maximum_length,
                      start_of_sequence_id=2, end_of_sequence_id=3, temperature=1.0,
                      name=None):

    with tf.name_scope(name, "gumbel_decoder_fn", [
            encoder_state, embedding_matrix, output_projections, maximum_length,
            start_of_sequence_id, end_of_sequence_id, temperature]) as scope:
        batch_size = tf.shape(encoder_state)[0]
        W, b = output_projections
        start_of_sequence_id = ops.convert_to_tensor(start_of_sequence_id, tf.int32)
        end_of_sequence_id = ops.convert_to_tensor(end_of_sequence_id, tf.int32)
        temperature = ops.convert_to_tensor(temperature, tf.float32)
        maximum_length = ops.convert_to_tensor(maximum_length, tf.int32)

    def _decoder_fn(time, cell_state, cell_input, cell_output, context_state):
        with ops.name_scope(name, "gumbel_decoder_fn",
                            [time, cell_state, cell_input, cell_output, context_state]):
            if cell_input is not None:
                raise ValueError("Expected cell_input to be None, but saw: %s" %
                                 cell_input)
            if cell_output is None:
                next_done = array_ops.zeros([batch_size, ], dtype=dtypes.bool)
                next_cell_state = encoder_state
                next_cell_input = tf.reshape(tf.tile(embedding_matrix[start_of_sequence_id], [batch_size]),
                                             shape=tf.shape(encoder_state))
                emit_output = cell_output
                # only next_context_state is returned from the rnn so save the next_cell_input in next_context_state
                next_context_state = next_cell_input  # context_state

            else:
                with ops.name_scope(name, "gumbel_output_fn", [W, b, cell_output, end_of_sequence_id, temperature]):
                    # -- output projection parameters usually used for output logits prior to softmax
                    output_logits = tf.add(tf.matmul(cell_output, W), b)  # [B, H] * [H, V] + [V] -> [B, V]

                    # -- stopping criterion if argmax is
                    output_argmax = tf.cast(tf.argmax(output_logits, axis=1), tf.int32)
                    next_done = tf.equal(output_argmax, end_of_sequence_id)

                    # -- sample from gumbel softmax (aka concrete) distribution, higher the temperature the spikier
                    output_probs = gumbel_softmax(output_logits, temperature=temperature, hard=False)

                    # soft embeddings for the next input
                    next_cell_input = tf.matmul(output_probs, embedding_matrix)  # [B, V] * [V, H] -> [B, H]

                next_cell_state = cell_state
                emit_output = cell_output
                # only next_context_state is returned from the rnn so save the next_cell_input in next_context_state
                next_context_state = next_cell_input  # context_state

            next_done = control_flow_ops.cond(math_ops.greater(time, maximum_length),
                                         lambda: array_ops.ones([batch_size, ], dtype=dtypes.bool),
                                         lambda: next_done)

        return next_done, next_cell_state, next_cell_input, emit_output, next_context_state

    return _decoder_fn

    # # -- output projection parameters usually used for output logits prior to softmax
    # W, b = output_projections
    # output_logits = tf.add(tf.matmul(cell_output, W), b)  # [B, H] * [H, V] + [V] -> [B, V]

    # # -- stopping criterion if argmax is
    # output_argmax = tf.argmax(output_logits, axis=1)
    # next_done = tf.equal(output_argmax, end_of_sequence_id)

    # # -- sample from gumbel softmax (aka concrete) distribution, higher the temperature the spikier
    # output_probs = gumbel_softmax(output_logits, temperature=temperature, hard=False)

    # # soft embeddings for the next input
    # next_input = tf.matmul(output_probs, embedding_matrix)  # [B, V] * [V, H] -> [B, H]
    # return next_input, next_done


# TODO: separate initial function and transition function
# TODO: minimize arguments
# def general_decoder_fn(
#         output_fn, init_fn, batch_size,
#         # encoder_state, embeddings,
#         # start_of_sequence_id, end_of_sequence_id,
#         # maximum_length, num_decoder_symbols,
#         dtype=dtypes.int32, name=None):
#     """ A more general form of decoder_fn that doens't force argmax.
#     Removed a few superfluous arguments.
#     """

#     with tf.name_scope(name, "general_decoder_fn",
#                        [output_fn, batch_size,
#                        # encoder_state, embeddings,
#                         # start_of_sequence_id, end_of_sequence_id,
#                         # maximum_length, num_decoder_symbols, 
#                         dtype]) as scope:
#         # start_of_sequence_id = ops.convert_to_tensor(start_of_sequence_id, dtype)
#         # end_of_sequence_id = ops.convert_to_tensor(end_of_sequence_id, dtype)
#         # maximum_length = ops.convert_to_tensor(maximum_length, dtype)
#         # num_decoder_symbols = ops.convert_to_tensor(num_decoder_symbols, dtype)
#         # encoder_info = nest.flatten(encoder_state)[0]
#         # batch_size = encoder_info.get_shape()[0].value
#         # if output_fn is None:
#             # output_fn = lambda x: x
#         # if batch_size is None:
#             # batch_size = array_ops.shape(encoder_info)[0]

#     def _decoder_fn(time, cell_state, cell_input, cell_output, context_state):
#         with ops.name_scope(name, "general_decoder_fn",
#                             [time, cell_state, cell_input, cell_output, context_state]):
#             if cell_input is not None:
#                 raise ValueError("Expected cell_input to be None, but saw: %s" %
#                                  cell_input)
#             # if cell_output is None:
#                 # -- initial values
#                 # next_done = array_ops.zeros([batch_size, ], dtype=dtypes.bool)
#                 # next_cell_state = encoder_state
#                 # next_cell_input = tf.reshape(tf.tile(embeddings[start_of_sequence_id], [batch_size]),
#                 #                              shape=tf.shape(encoder_state))
#                 # emit_output = cell_output
#                 # # only next_context_state is returned from the rnn so save the next_cell_input in next_context_state
#                 # next_context_state = (next_cell_input, )  # context_state
#                 # next_done, next_cell_state, next_cell_input, emit_output, next_context_state = \
#                     # init_fn(time, cell_state, cell_input, cell_output, context_state)

#             # else:
#                 # -- if no stopping criterion
#                 # next_done = array_ops.zeros([batch_size, ], dtype=dtypes.bool)
#                 # next_cell_state = cell_state
#                 # next_cell_input, next_done = output_fn(cell_output)
#                 # emit_output = cell_output
#                 # only next_context_state is returned from the rnn so save the next_cell_input in next_context_state
#                 # next_context_state = (next_cell_input, )  # context_state
#             next_done, next_cell_state, next_cell_input, emit_output, next_context_state = \
#                 output_fn(time, cell_state, cell_input, cell_output, context_state)

#             # if time > maxlen, return all true vector
#             next_done = control_flow_ops.cond(math_ops.greater(time, maximum_length),
#                                          lambda: array_ops.ones([batch_size, ], dtype=dtypes.bool),
#                                          lambda: next_done)

#             return next_done, next_cell_state, next_cell_input, emit_output, next_context_state

#     return _decoder_fn


if __name__ == "__main__":
    import numpy as np
    from tensorflow.contrib import seq2seq
    from tensorflow.contrib import rnn

    sequence_length = 25
    vocab_size = 100
    hidden_dims = 10
    batch_size = 32

    cell = rnn.BasicRNNCell(hidden_dims)
    encoder_state = tf.constant(np.random.randn(batch_size, hidden_dims), dtype=np.float32)
    embeddings = tf.constant(np.random.randn(vocab_size, hidden_dims), dtype=np.float32)
    output_W = tf.transpose(embeddings)  # -- tied embeddings
    output_b = tf.constant(np.random.randn(vocab_size), dtype=np.float32)
    output_projections = (output_W, output_b)
    maximum_length=tf.reduce_max(sequence_length) + 3
    # output_fn = functools.partial(gumbel_output_fn, output_projections=output_projections)

    # decoder_fn = general_decoder_fn(output_fn, encoder_state, embeddings, 
        # start_of_sequence_id=2, end_of_sequence_id=3, 
        # maximum_length=tf.reduce_max(sequence_length) + 3, num_decoder_symbols=vocab_size)

    decoder_fn = gumbel_decoder_fn(encoder_state, embeddings, output_projections, maximum_length)

    outputs, final_state, final_context_state = seq2seq.dynamic_rnn_decoder(
            cell, decoder_fn, inputs=None, sequence_length=sequence_length)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        outputs_ = sess.run(outputs)        
        print("outputs shape", outputs_.shape)
