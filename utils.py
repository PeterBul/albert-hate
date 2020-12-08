from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow.compat.v1 as tf   #pylint: disable=import-error
import os
import sentencepiece as spm
from constants import class_probabilities, num_labels

def read_tfrecord_builder(is_training, seq_length, regression=False):
    def read_tfrecord(serialized_example):
        """
        Map output of TFRecord which only has 1-D tensors to multidimensional tensors, and give input from dataset in proper format.
        Each example comes with several possible captions, so choose one randomly. This can be changed to using a non-probabilistic method.
        """
        feature_description = {
          "input_ids": tf.FixedLenFeature([128], tf.int64),
          "input_mask": tf.FixedLenFeature([128], tf.int64),
          "segment_ids": tf.FixedLenFeature([128], tf.int64),
          "is_real_example": tf.FixedLenFeature([], tf.int64),
        }

        if regression and is_training:
            feature_description["average"] = tf.FixedLenFeature([], tf.float32)
        else:
            feature_description["label_ids"] = tf.FixedLenFeature([], tf.int64)
      
        example = decode_record(serialized_example, feature_description)

        if seq_length < 128:
          for key in ['input_ids', 'input_mask', 'segment_ids']:
            example[key] = example[key][:seq_length]
            tf.logging.info(example[key])
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


def get_sentence_piece_processor():
    return spm.SentencePieceProcessor(model_file='albert_base' + os.sep + '30k-clean.model')               # pylint: disable=unexpected-keyword-arg


def oversample_classes(example, dataset, oversampling_coef=0.9):
  """
  Returns the number of copies of given example
  """
  label_id = example['label_ids']
  # Fn returning negative class probability
  #def f1(i): return tf.constant(class_probabilities[args.dataset][i])
  #def f2(): return tf.cond(tf.math.equal(label_id, tf.constant(1)), lambda: f1(1), lambda: f1(2))

  #class_prob = tf.cond(tf.math.equal(label_id, tf.constant(0)), lambda: f1(0), f2)
  class_prob = tf.gather(class_probabilities[dataset], label_id)
  class_target_prob = tf.constant(1/num_labels[dataset])
  #class_target_prob = tf.reduce_max(class_probabilities[dataset])
  #class_target_prob = example['class_target_prob']
  prob_ratio = tf.cast(class_target_prob/class_prob, dtype=tf.float32)
  # soften ratio is oversampling_coef==0 we recover original distribution
  prob_ratio = prob_ratio ** oversampling_coef 
  # for classes with probability higher than class_target_prob we
  # want to return 1
  prob_ratio = tf.maximum(prob_ratio, 1) 
  # for low probability classes this number will be very large
  repeat_count = tf.floor(prob_ratio)
  # prob_ratio can be e.g 1.9 which means that there is still 90%
  # of change that we should return 2 instead of 1
  repeat_residual = prob_ratio - repeat_count # a number between 0-1
  residual_acceptance = tf.less_equal(
                      tf.random_uniform([], dtype=tf.float32), repeat_residual
  )

  residual_acceptance = tf.cast(residual_acceptance, tf.int64)
  repeat_count = tf.cast(repeat_count, dtype=tf.int64)

  #tf.logging.info("Oversampling label with repeat count: " + str(repeat_count + residual_acceptance))

  return repeat_count + residual_acceptance

def deterministic_oversampling(example):
  label_id = example['label_ids']
  if label_id == 0:
    return 4
  return 1



def undersampling_filter(example, dataset, undersampling_coef=0.9):
  """
  Computes if given example is rejected or not.
  """
  label_id = example['label_ids']
  # Fn returning negative class probability
  #def f1(i): return tf.constant(class_probabilities[args.dataset][i])
  #def f2(): return tf.cond(tf.math.equal(label_id, tf.constant(1)), lambda: f1(1), lambda: f1(2))

  #class_prob = tf.cond(tf.math.equal(label_id, tf.constant(0)), lambda: f1(0), f2)
  class_prob = tf.gather(class_probabilities[dataset], label_id)
  class_target_prob = tf.constant(1/num_labels[dataset])

  prob_ratio = tf.cast(class_target_prob/class_prob, dtype=tf.float32)
  prob_ratio = prob_ratio ** undersampling_coef
  prob_ratio = tf.minimum(prob_ratio, 1.0)

  acceptance = tf.less_equal(tf.random_uniform([], dtype=tf.float32), prob_ratio)
  # predicate must return a scalar boolean tensor
  return acceptance