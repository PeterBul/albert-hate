import tensorflow as tf
import utils
import os
import re
import sentencepiece as spm
from paths import SOLID_CONVERTED, SOLID_CONVERTED_TEST, OLID_CONVERTED

tf.enable_eager_execution()
def get_sentence_piece_processor():
  return spm.SentencePieceProcessor(model_file='albert_base' + os.sep + '30k-clean.model')

def shard(ds):
  ds = ds.shard(10, 1)
  tf.random.set_random_seed(42)
  ds = ds.flat_map(
              lambda x, y: tf.data.Dataset.from_tensor_slices((x,y)).repeat(utils.deterministic_oversampling(x, ernie=True))
          )
  ds = ds.filter(lambda x: utils.deterministic_undersampling_filter(x, ernie=True))
  ds = ds.shuffle(2048, seed=10)
  return ds

def get_train():
  ds = tf.data.TFRecordDataset(SOLID_CONVERTED)
  ds = ds.shard(10, 1)
  tf.random.set_random_seed(42)
  ds = ds.shuffle(2048, seed=10)
  ds = ds.map(utils.read_tfrecord_builder(is_training=False, seq_length=128))
  ds = ds.flat_map(
              lambda x: tf.data.Dataset.from_tensors(x).repeat(utils.deterministic_oversampling(x))
          )
  ds = ds.filter(lambda x: utils.deterministic_undersampling_filter(x))
  return ds

def get_ernie_dataset():
  ds = tf.data.experimental.CsvDataset(
    '../ernie/data/solid/conv/train.tsv', [tf.string, tf.int32],
    header=True, field_delim='\t')
  return ds


def get_test():
  olid_train = tf.data.TFRecordDataset(OLID_CONVERTED)
  solid_test = tf.data.TFRecordDataset(SOLID_CONVERTED_TEST)
  ds = olid_train.concatenate(solid_test)
  ds = ds.shard(10, 1)
  ds = ds.map(utils.read_tfrecord_builder(is_training=False, seq_length=128))
  return ds

def decode(input_ids, sp):
  text = ''.join(list([sp.id_to_piece(int(id)) for id in input_ids])).replace('▁', ' ')
  text = re.sub(r'(<pad>)*$|(\[SEP\])|^(\[CLS\])', '', text)
  return text



ds = get_ernie_dataset()
ds = shard(ds)


i = 0
counter = [0,0,0]
for elem in ds:
  counter[elem[1].numpy()] += 1
  if i > 10000:
    break
  i += 1
print(counter)



