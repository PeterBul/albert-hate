import tensorflow as tf
import utils
import os
import numpy as np
from constants import num_labels
from contextlib import ExitStack

def count_labels(dataset_name, use_oversampling, use_undersampling):
  iterator = load_iterator(dataset_name, use_oversampling, use_undersampling)
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

def count_examples(dataset_name, use_oversampling, use_undersampling, use_cpu=False):

  with ExitStack() as stack:
    if use_cpu:
      stack.enter_context(tf.device('/cpu:0'))
     

    iterator = load_iterator(dataset_name, use_oversampling, use_undersampling)
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

def load_iterator(dataset_name, use_oversampling, use_undersampling):
  tfrecord = os.path.join('data', 'tfrecords', dataset_name, 'train-128.tfrecords')
  dataset = tf.data.TFRecordDataset(tfrecord)
  dataset = dataset.map(utils.read_tfrecord_builder(is_training=True, seq_length=128, regression=False))
  if use_oversampling:
    dataset = dataset.flat_map(
                lambda x: tf.data.Dataset.from_tensors(x).repeat(utils.oversample_classes(x, dataset_name))
            )
  if use_undersampling:
    dataset = dataset.filter(utils.undersampling_filter)

  iterator = dataset.make_one_shot_iterator()

  return iterator

if __name__ == "__main__":
  import argparse
  parser = argparse.ArgumentParser()
  parser.add_argument('--use_cpu', default=False, action='store_true')
  args = parser.parse_args()
  print(count_examples('founta', use_oversampling=True, use_undersampling=False, use_cpu=args.use_cpu))