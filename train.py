""" Train
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import json
import tqdm
from contextlib import contextmanager

import tensorflow as tf

from utils import start_threads, set_logging_verbosity
from data_loader import DATA_PATH, queue_context
from model import Model

flags = tf.flags

# -- saver options
flags.DEFINE_string("model_dir", "./tmp", (
    "Model directory."))

# -- train options
flags.DEFINE_string("logging_verbosity", "INFO", (
    "Set verbosity to INFO, WARN, DEBUG or ERROR"))
flags.DEFINE_string("corpus_name", "ptb", (
    "Corpus name."))
flags.DEFINE_integer("batch_size", 32, (
    "Batch size for dequeue."))
flags.DEFINE_integer("epoch_size", 10, (
    "Max epochs."))

# -- optimizer options
flags.DEFINE_float("learning_rate", 0.005, (
    "Learning rate for optimizer."))
flags.DEFINE_float("max_grads", 5.0, (
    "Max clipping of gradients."))

# -- model options
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
opts = FLAGS.__flags  # dict TODO: make class?

set_logging_verbosity(FLAGS.logging_verbosity)


def _get_n_batches(batch_size, corpus_size):
    return int(corpus_size // batch_size)


def set_initial_ops():
    local_init_op = tf.local_variables_initializer()
    global_init_op = tf.global_variables_initializer()
    init_op = tf.group(local_init_op, global_init_op)
    return init_op


def set_train_op(loss, **opts):
    optimizer = tf.train.AdamOptimizer(learning_rate=opts["learning_rate"])

    gradients = optimizer.compute_gradients(loss)
    clipped_gradients = [(grad if grad is None else tf.clip_by_norm(grad, opts["max_grads"]), var) 
        for grad, var in gradients]

    train_op = optimizer.apply_gradients(clipped_gradients)
    return train_op


def get_supervisor(model, **opts):
    saver = tf.train.Saver()
    summary_writer = tf.summary.FileWriter(opts["model_dir"])

    supervisor = tf.train.Supervisor(
        logdir=opts["model_dir"],
        is_chief=True,
        saver=saver,
        # init_op=model.init_op,
        # summary_op=model.summary_op,
        summary_writer=summary_writer,
        save_summaries_secs=100,  # TODO: add as flags
        save_model_secs=100,
        global_step=model.global_step)

    return supervisor


def get_sess_config(**opts):
    # gpu_options = tf.GPUOptions(
        # per_process_gpu_memory_fraction=self.gpu_memory_fraction,
        # allow_growth=True) # seems to be not working

    sess_config = tf.ConfigProto(
        # log_device_placement=True,
        inter_op_parallelism_threads=8, 
        # allow_soft_placement=True,
        # gpu_options=gpu_options)
        )

    return sess_config


def main():
    # TODO: change opts to flags
    path = DATA_PATH[FLAGS.corpus_name]["train"]
    model = Model(path, **opts)

    n_batches = _get_n_batches(model.opts["batch_size"], model.corpus_size)

    g_loss = model.g_tensors_pretrain.loss
    g_train_op = set_train_op(g_loss, **opts)

    d_loss_real = model.d_tensors_real.loss
    d_loss_generated = model.d_tensors_generated.loss
    d_loss = (tf.reduce_mean(d_loss_real) + tf.reduce_mean(d_loss_generated)) / 2.0
    d_train_op = set_train_op(d_loss, **opts)

    init_op = set_initial_ops()

    # TODO: restore global_step from saved ckpt file?
    sv = get_supervisor(model, **opts)
    sess_config = get_sess_config(**opts)

    with sv.managed_session(config=sess_config) as sess:
        # sess.run(init_op)  # TODO: managed_session seems to handle this ok

        threads = start_threads(model.enqueue_data, (sess, ))

        # TODO: add logging of cost as callback to supervisor
        def print_loss(sess):
            _g, _d = sess.run([g_loss, d_loss])
            tf.logging.info("g_loss: %.4f, d_loss: %.4f", _g, _d)
        sv.loop(60, print_loss, (sess, ))

        """
        WARNING:tensorflow:Error encountered when serializing rnn_cell.
        Type is unsupported, or the types of the items don't match field type in CollectionDef.
        'DropoutWrapper' object has no attribute 'name'
        """
        # with queue_context(sess):  # TODO: managed_session seems to handle this ok
        for _ in tqdm.tqdm(range(n_batches * opts["epoch_size"])):
            if sv.should_stop():
                break
            # TODO: add learning rate decay
            sess.run([g_train_op, d_train_op, model.global_step])


if __name__ == "__main__":
    main()
