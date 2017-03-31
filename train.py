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

FLAGS = flags.FLAGS


def set_train_op(loss, **opts):
    cost = tf.reduce_mean(loss)

    optim = tf.train.AdamOptimizer(learning_rate=0.005)
    train_op = optim.minimize(cost)
    return train_op


if __name__ == "__main__":
    path = DATA_PATH[FLAGS.corpus_name]["train"]
    model = Model(path)

    loss = model.g_tensors_pretrain.loss
    train_op = set_train_op(loss)

    with tf.Session() as sess:
        sess.run(tf.local_variables_initializer())
        sess.run(tf.global_variables_initializer())

        start_threads(model.enqueue_data, (sess, ))

        with queue_context(sess):
            epoch_size = 10000
            for _ in tqdm.tqdm(range(epoch_size)):
                sess.run(train_op)
