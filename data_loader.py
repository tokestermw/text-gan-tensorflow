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
        "valid": os.path.join(DATA_DIR, "ptb", "valid.txt"),
        "test": os.path.join(DATA_DIR, "ptb", "test.txt"),
        "vocab": os.path.join(DATA_DIR, "ptb", "vocab.pkl"),
    }
}

SPECIAL_TOKENS = { "_PAD": 0, "_OOV": 1, "_START": 2, "_END": 3}

MAXLEN = 100  # maximum words in a line


# TODO: spacy tokenizer
def tokenize(line):
    tokens = line.split()
    return tokens[:MAXLEN]


def read_data(path):
    with open(path, 'r') as f:
        for line in f:
            line = line.strip()
            yield line


# TODO: change so you can change the save_path
@maybe_save(save_path=DATA_PATH["ptb"]["vocab"])
def build_vocab(path, min_counts=10):
    counts = Counter()

    corpus_size = 0
    for line in read_data(path):
        corpus_size += 1
        tokens = tokenize(line)
        for token in tokens:
            counts[token] += 1

    word2idx = {word: idx + len(SPECIAL_TOKENS) for idx, (word, count) in enumerate(counts.most_common()) if count > min_counts}
    word2idx.update(SPECIAL_TOKENS)

    idx2word = {idx: word for word, idx in word2idx.items()}

    return word2idx, idx2word, corpus_size


def vectorize(line, word2idx):
    tokens = tokenize(line)
    vector = [word2idx.get(token, SPECIAL_TOKENS["_OOV"]) for token in tokens]
    vector = [SPECIAL_TOKENS["_START"]] + vector + [SPECIAL_TOKENS["_END"]]
    return vector


def preprocess(data):
    # PaddingFIFOQueue pads to the max size seen in the data (instead of the minibatch)
    # by chopping off the ends, this limits redundant computations in the output layer
    sequence_length = tf.reduce_sum(tf.cast(tf.not_equal(data, 0), dtype=tf.int32), axis=1)
    maximum_sequence_length = tf.reduce_max(sequence_length)
    data = data[:, :maximum_sequence_length] 

    source = data[:, :-1]
    target = data[:, 1:]
    sequence_length -= 1
    return source, target, sequence_length


def get_input_queues(path, word2idx, batch_size=32, num_threads=8):
    input_ph = tf.placeholder(tf.int32, shape=[None])  # [B, T]
    queue = tf.PaddingFIFOQueue(shapes=[[None, ]], dtypes=[tf.int32], capacity=5000,)

    # TODO: enqueue_many would be faster, would require batch and padding at numpy-level
    enqueue_op = queue.enqueue([input_ph])
    def enqueue_data(sess):
        # for epoch in range(epoch_size):
        while True:  # 
            for idx, line in enumerate(read_data(path)):
                v = vectorize(line, word2idx)
                sess.run(enqueue_op, feed_dict={input_ph: v})

    # dequeue_batch = queue.dequeue_many(batch_size)
    dequeue_op = queue.dequeue()
    dequeue_batch = tf.train.batch([dequeue_op], batch_size=batch_size, num_threads=num_threads, capacity=1000, 
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


def _check_for_duplicates(word_ids, batch_size):
    word_ids = map(tuple, word_ids)
    dupes = set(word_ids)
    return batch_size - len(dupes)


if __name__ == "__main__":
    path = DATA_PATH["ptb"]["train"]
    word2idx, idx2word, corpus_size = build_vocab(path)

    with tf.Session() as sess:
        batch_size = 32
        enqueue_data, dequeue_batch = get_input_queues(path, word2idx, batch_size=batch_size)
        threads = start_threads(enqueue_data, (sess, ))

        with queue_context(sess):
            while True:
                source, target, sequence_length = preprocess(dequeue_batch)
                s, t, l = sess.run([source, target, sequence_length])
                print("dupes", _check_for_duplicates(s.tolist(), batch_size))
                print(s.shape, t.shape, l.shape)
                # for _s in s.tolist():
                    # print([idx2word[i] for i in _s])
                # for _t in t.tolist():
                    # print([idx2word[i] for i in _t])
