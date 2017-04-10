""" Train
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import json
import tqdm
from contextlib import contextmanager

import tensorflow as tf

from utils import start_threads, set_logging_verbosity, MovingAverage, count_number_of_parameters
from data_loader import DATA_PATH, queue_context, tokenize, vectorize
from layers import _phase_train, _phase_infer
from model import Model
from search import reverse_decode, greedy_argmax

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
flags.DEFINE_string("seed_text", "how are", (
    "Seed the sampling from the generator with this text."))
flags.DEFINE_string("gan_strategy", "generator", (
    "GAN training strategy (generator, discriminator, both)."))

# -- optimizer options
flags.DEFINE_float("learning_rate", 0.0005, (
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
flags.DEFINE_float("output_dropout_keep_prob", 0.5, (
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


def set_train_op(loss):
    optimizer = tf.train.AdamOptimizer(learning_rate=FLAGS.learning_rate)
    # optimizer = tf.train.GradientDescentOptimizer(0.01)

    gradients = optimizer.compute_gradients(loss)
    clipped_gradients = [(grad if grad is None else tf.clip_by_norm(grad, FLAGS.max_grads), var) 
        for grad, var in gradients]

    train_op = optimizer.apply_gradients(clipped_gradients)
    return train_op


def get_supervisor(model):
    saver = tf.train.Saver()
    summary_writer = tf.summary.FileWriter(FLAGS.model_dir)

    supervisor = tf.train.Supervisor(
        logdir=FLAGS.model_dir,
        is_chief=True,
        saver=saver,
        init_op=set_initial_ops(),
        summary_op=tf.summary.merge_all(),
        summary_writer=summary_writer,
        save_summaries_secs=100,  # TODO: add as flags
        save_model_secs=1000,
        global_step=model.global_step,
        )

    return supervisor


def get_sess_config():
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


def print_loss(sess, loss, moving_average=None):
    l = sess.run(loss)
    if moving_average is None:
        tf.logging.info(" loss: %.4f", l)
    else:
        l_ma = moving_average.next(l)
        tf.logging.info(" loss: %.4f", l_ma)

    # _g, _d = sess.run([g_loss, d_loss])
    # tf.logging.info("g_loss: %.4f, d_loss: %.4f", _g, _d)


# TODO: add to TensorBoard
def print_valid_loss(sess, loss):
    sess.run(_phase_infer)

    total_loss = 0.0
    for _ in range(100):  # TODO: change, use all test data
        l = sess.run(loss)
        total_loss += l

    valid_loss = total_loss / 100.
    tf.logging.info(" valid_loss: %.4f", valid_loss)

    sess.run(_phase_train)


# TODO: configurable seed_text
def print_sample(sess, seed_text, probs, input_ph, word2idx, idx2word):
    # seed_text = "how are you"
    vector = vectorize(seed_text, word2idx)
    out = greedy_argmax(vector[:-1], lambda x: sess.run(probs, {input_ph: [x]}))
    text = reverse_decode(out, idx2word)
    tf.logging.info(" generated text:\n%s", text)


# TODO: learing rate decay
def main():
    corpus = DATA_PATH[FLAGS.corpus_name]
    # TODO: move to flags
    model = Model(corpus, **opts)

    n_batches = _get_n_batches(FLAGS.batch_size, model.corpus_size)

    g_loss = model.g_tensors_pretrain.loss
    g_train_op = set_train_op(g_loss)

    g_loss_valid = model.g_tensors_pretrain_valid.loss

    d_loss_real = model.d_tensors_real.loss
    d_loss_generated = model.d_tensors_generated.loss
    d_loss = (tf.reduce_mean(d_loss_real) + tf.reduce_mean(d_loss_generated)) / 2.0
    d_train_op = set_train_op(d_loss)

    g_loss_ma = MovingAverage(10)

    sv = get_supervisor(model)
    sess_config = get_sess_config()

    tf.logging.info(" number of parameters %i", count_number_of_parameters())

    with sv.managed_session(config=sess_config) as sess:
        sess.run(_phase_train)

        threads = start_threads(model.enqueue_data, (sess, ))
        threads_valid = start_threads(model.enqueue_data_valid, (sess, ))

        # TODO: add learning rate decay -> early_stop
        sv.loop(60, print_loss, (sess, g_loss, g_loss_ma))
        sv.loop(600, print_valid_loss, (sess, g_loss_valid))
        sv.loop(100, print_sample, (sess, FLAGS.seed_text, model.g_tensors_pretrain_valid.flat_logits, 
            model.input_ph, model.word2idx, model.idx2word))  # TODO: cleanup

        # make graph read only
        sess.graph.finalize()

        for epoch in range(FLAGS.epoch_size):
            tf.logging.info(" epoch: %i", epoch)

            for _ in tqdm.tqdm(range(n_batches)):
                if sv.should_stop():
                    break

                # TODO: add strategies
                # print(sess.run(model.source_valid))
                # print(sess.run(g_loss_valid))
                sess.run([g_train_op, model.increment_global_step_op])  # only run generator
                # sess.run([g_train_op, d_train_op, model.global_step])

                if False:
                    # some criterion
                    sv.stop()

        sv.saver.save(sess, sv.save_path, global_step=sv.global_step)
        tf.logging.info(" training finished")


if __name__ == "__main__":
    main()
