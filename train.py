""" Train
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tqdm

import tensorflow as tf

from utils import start_threads
from data_loader import DATA_PATH, queue_context
from model import Model

flags = tf.flags
logging = tf.logging

logging.set_verbosity(logging.INFO)

flags.DEFINE_string("corpus_name", "ptb", (
    "Corpus name."))
flags.DEFINE_integer("batch_size", 32, (
    "Batch size for dequeue."))
flags.DEFINE_integer("embedding_dim", 128, (
    "Hidden dimensions for embedding."))
flags.DEFINE_integer("rnn_hidden_dim", 128, (
    "Hidden dimensions for RNN hidden vectors."))
flags.DEFINE_integer("output_hidden_dim", 128, (
    "Hidden dimensions for output hidden vectors before softmax layer."))
flags.DEFINE_float("word_dropout_keep_prob", 0.9, (
    "Dropout keep rate for word embeddings."))
flags.DEFINE_float("recurrent_dropout_keep_prob", 0.6, (
    "Dropout keep rate for recurrent input and output vectors."))
flags.DEFINE_float("output_dropout_keep_prob", 0.8, (
    "Dropout keep rate for output vectors."))

FLAGS = flags.FLAGS
opts = FLAGS.__flags  # dict


def set_initial_ops():
    local_init_op = tf.local_variables_initializer()
    global_init_op = tf.global_variables_initializer()
    return local_init_op, global_init_op


def set_train_op(loss, **opts):
    cost = tf.reduce_mean(loss)
    optim = tf.train.AdamOptimizer(learning_rate=0.005)
    train_op = optim.minimize(cost)
    return train_op


def main():
    path = DATA_PATH[FLAGS.corpus_name]["train"]
    model = Model(path, **opts)

    g_loss = model.g_tensors_pretrain.loss
    g_train_op = set_train_op(g_loss)

    d_loss_real = model.d_tensors_real.loss
    d_loss_generated = model.d_tensors_generated.loss
    d_loss = tf.reduce_mean(d_loss_real) + tf.reduce_mean(d_loss_generated)
    d_train_op = set_train_op(d_loss)

    local_init_op, global_init_op = set_initial_ops()

    with tf.Session() as sess:
        sess.run([local_init_op, global_init_op])

        threads = start_threads(model.enqueue_data, (sess, ))

        with queue_context(sess):
            epoch_size = 10000
            for _ in tqdm.tqdm(range(epoch_size)):
                sess.run(g_train_op)
                sess.run(d_train_op)


if __name__ == "__main__":
    main()
