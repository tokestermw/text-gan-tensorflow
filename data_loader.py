""" Batch loader of PTB data.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
from collections import Counter
from contextlib import contextmanager

import numpy as np
import tensorflow as tf

# -- local imports
from utils import maybe_save, start_threads


DATA_DIR = "data"
DATA_PATH = {
    "ptb": {
        "train": os.path.join(DATA_DIR, "ptb", "train.txt"),
        "test": os.path.join(DATA_DIR, "ptb", "test.txt"),
        "valid": os.path.join(DATA_DIR, "ptb", "valid.txt"),
        "vocab": os.path.join(DATA_DIR, "ptb", "vocab.pkl"),
    }
}


# TODO: spacy tokenizer
def tokenize(line):
    return line.split()


def read_data(path):
    with open(path, 'r') as f:
        for line in f:
            line = line.strip()
            yield line


# TODO: make save_path a command line option
# TODO: add padding, oov, start, end symbols to vocab
@maybe_save(save_path=DATA_PATH["ptb"]["vocab"])
def build_vocab(path):
    counts = Counter()

    for line in read_data(path):
        tokens = tokenize(line)
        for token in tokens:
            counts[token] += 1

    word2idx = {word: idx for idx, (word, count) in enumerate(counts.most_common())}
    idx2word = {idx: word for word, idx in word2idx.items()}

    return word2idx, idx2word


def vectorize(line, word2idx):
    tokens = tokenize(line)
    vector = [word2idx[token] for token in tokens]
    return vector


# TODO: get maximum sequence_length and limit it
def preprocess(data):
    source = data[:, :-1]
    target = data[:, 1:]
    sequence_length = tf.reduce_sum(tf.cast(tf.not_equal(target, 0), dtype=tf.int32), axis=1)
    return source, target, sequence_length


def get_and_run_input_queues(path, word2idx, batch_size=32):
    input_ph = tf.placeholder(tf.int32, shape=[None])  # [B, T]
    queue = tf.PaddingFIFOQueue(shapes=[[None, ]], dtypes=[tf.int32], capacity=5000,)

    # TODO: enqueue_many would be faster, would require batch and padding at numpy-level
    enqueue_op = queue.enqueue([input_ph])
    def enqueue_data(sess, epoch_size=10):
        for epoch in range(epoch_size):
            for idx, line in enumerate(read_data(path)):
                v = vectorize(line, word2idx)
                sess.run(enqueue_op, feed_dict={input_ph: v})

    # queue_runner = tf.train.QueueRunner(queue, [enqueue_op] * 8)
    # tf.train.add_queue_runner(queue_runner)

    dequeue_op = queue.dequeue()
    dequeue_batch = tf.train.batch([dequeue_op], batch_size=batch_size, capacity=1000, 
        dynamic_pad=True, name="batch_and_pad")

    return enqueue_data, dequeue_batch


@contextmanager
def queue_context(sess):
    # thread coordinator
    coord = tf.train.Coordinator()
    try:
        # start queue thread
        threads = tf.train.start_queue_runners(sess, coord)
        yield
    except tf.errors.OutOfRangeError:
        print("Done training")
    except KeyboardInterrupt:
        print("Force stop.")
    finally:
        # stop queue thread
        coord.request_stop()
        # wait thread to exit.
        coord.join(threads)


if __name__ == "__main__":
    path = DATA_PATH["ptb"]["train"]
    word2idx, idx2word = build_vocab(path)

    with tf.Session() as sess:
        enqueue_data, dequeue_batch = get_and_run_input_queues(path, word2idx)
        threads = start_threads(enqueue_data, (sess, ))

        with queue_context(sess):
            while True:
                source, target, sequence_length = preprocess(dequeue_batch)
                s, t, l = sess.run([source, target, sequence_length])
                print(s.shape, t.shape, l.shape)
