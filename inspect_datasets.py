import tensorflow as tf
import utils
import os
import re
import numpy as np
from constants import num_labels
from contextlib import ExitStack
from paths import SOLID_CONVERTED, SOLID_CONVERTED_TEST, OLID_CONVERTED
import sentencepiece as spm

ALBERT_PRETRAINED_PATH = 'albert_' + 'base'

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
  print(count)
  print(count.sum())
  count = count/count.sum()
  

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

def get_sentence_piece_processor():
  return spm.SentencePieceProcessor(model_file=ALBERT_PRETRAINED_PATH + os.sep + '30k-clean.model')

def inspect():
  tf.enable_eager_execution()
  sp = get_sentence_piece_processor()
  ds = tf.data.TFRecordDataset(OLID_CONVERTED)
  ds = ds.map(utils.read_tfrecord_builder(is_training=False, seq_length=128))
  i = 0
  for elem in ds:
    input_ids = elem['input_ids'].numpy()
    text = decode(input_ids, sp)
    print(text)
    if i > 2:
      break
    i += 1
  

def get_train():
  ds = tf.data.TFRecordDataset(SOLID_CONVERTED)
  ds = ds.shard(10, 1)
  tf.random.set_random_seed(42)
  ds = ds.map(utils.read_tfrecord_builder(is_training=False, seq_length=128))
  ds = ds.flat_map(
              lambda x: tf.data.Dataset.from_tensors(x).repeat(utils.deterministic_oversampling(x))
          )
  ds = ds.filter(lambda x: utils.deterministic_undersampling_filter(x))
  ds = ds.shuffle(2048, seed=10)
  return ds

def decode(input_ids, sp):
  text = ''.join(list([sp.id_to_piece(int(id)) for id in input_ids])).replace('▁', ' ')
  text = re.sub(r'(<pad>)*$|(\[SEP\])|^(\[CLS\])', '', text)
  return text

def load_iterator(dataset_name, use_oversampling, use_undersampling):
  tfrecord = os.path.join('data', 'tfrecords', dataset_name, 'dev-128.tfrecords' if dataset_name == 'converted' else 'train-128.tfrecords')
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
  #import argparse
  #parser = argparse.ArgumentParser()
  #parser.add_argument('--use_cpu', default=False, action='store_true')
  #args = parser.parse_args()
  #print(count_labels('converted', use_oversampling=False, use_undersampling=False))
  #print(count_examples('con', use_oversampling=True, use_undersampling=False, use_cpu=args.use_cpu))
  inspect()