from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow.compat.v1 as tf


def eval_input_fn(batch_size, test=False):
  if not test:
    tf.logging.info('Using DEV dataset: {}'.format(FILE_DEV))
    ds = tf.data.TFRecordDataset(FILE_DEV)
  else:
    tf.logging.info('Using TEST dataset: {}'.format(FILE_TEST))
    ds = tf.data.TFRecordDataset(FILE_TEST)
  ds = ds.map(read_tfrecord_builder(is_training=False)).batch(batch_size)
  return ds

if __name__ == "__main__":
  
  predict(
    input_fn, predict_keys=None, hooks=None, checkpoint_path=None,
    yield_single_examples=True
)