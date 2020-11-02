import tensorflow as tf
import utils
import os
import numpy as np
from constants import num_labels
import argparse
from contextlib import ExitStack



parser = argparse.ArgumentParser()
parser.add_argument('--use_cpu', default=False, action='store_true')
args = parser.parse_args()

def count_labels(dataset_name):
  iterator = load_iterator(dataset_name)
  next_element = iterator.get_next()

  count = [0 for i in range(num_labels[dataset_name])]
  with tf.compat.v1.Session() as sess:
    try:
      while True:
        e = sess.run(next_element)
        count[e['label_ids']] += 1
    except tf.errors.OutOfRangeError:
      pass

  count = np.asarray(count)

  return count

def count_examples(dataset_name):

  with ExitStack() as stack:
    if args.use_cpu:
      stack.enter_context(tf.device('/cpu:0'))
     

    iterator = load_iterator(dataset_name)
    next_element = iterator.get_next()

    tf.logging.info("Counting examples in dataset: {}".format(dataset_name))
    count = 0
    with tf.compat.v1.Session() as sess:
      try:
        while True:
          sess.run(next_element)
          count += 1
      except tf.errors.OutOfRangeError:
        pass
    tf.logging.info("Count: {}".format(count))

  return count

def load_iterator(dataset_name):
  tfrecord = os.path.join('data', 'tfrecords', dataset_name, 'train-128.tfrecords')
  dataset = tf.data.TFRecordDataset(tfrecord)
  dataset = dataset.map(utils.read_tfrecord_builder(is_training=True, seq_length=128, regression=False))
  dataset = dataset.flat_map(
              lambda x: tf.data.Dataset.from_tensors(x).repeat(utils.oversample_classes(x, dataset_name))
          )

  iterator = dataset.make_one_shot_iterator()

  return iterator

if __name__ == "__main__":
  print(count_examples('founta'))