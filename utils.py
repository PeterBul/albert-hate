from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow.compat.v1 as tf

def read_tfrecord_builder(is_training, seq_length, regression=False):
    def read_tfrecord(serialized_example):
        """
        Map output of TFRecord which only has 1-D tensors to multidimensional tensors, and give input from dataset in proper format.
        Each example comes with several possible captions, so choose one randomly. This can be changed to using a non-probabilistic method.
        """
        feature_description = {
          "input_ids": tf.FixedLenFeature([seq_length], tf.int64),
          "input_mask": tf.FixedLenFeature([seq_length], tf.int64),
          "segment_ids": tf.FixedLenFeature([seq_length], tf.int64),
          "is_real_example": tf.FixedLenFeature([], tf.int64),
        }

        if regression and is_training:
            feature_description["average"] = tf.FixedLenFeature([], tf.float32)
        else:
            feature_description["label_ids"] = tf.FixedLenFeature([], tf.int64)
      
        example = decode_record(serialized_example, feature_description)
        
        return example
    return read_tfrecord

def decode_record(record, feature_description):
    """Decodes a record to a TensorFlow example."""
    example = tf.parse_single_example(record, feature_description)

    # tf.Example only supports tf.int64, but the TPU only supports tf.int32.
    # So cast all int64 to int32.
    for name in list(example.keys()):
      t = example[name]
      if t.dtype == tf.int64:
        t = tf.to_int32(t)
      example[name] = t

    return example

def str2bool(value):
  if value is None:
    return value
  if isinstance(value, bool):
    return value
  if value.lower() in ('true', 't', '1'):
    return True
  if value.lower() in ('false', 'f', '0'):
    return False
  raise TypeError("Boolean value expected")