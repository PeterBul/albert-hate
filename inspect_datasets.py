import tensorflow as tf
import utils
import os
import numpy as np

FILE_TRAIN = os.path.join('data', 'tfrecords', 'founta', 'train-128.tfrecords')

with tf.device('/cpu:0'):
  dataset = tf.data.TFRecordDataset(FILE_TRAIN)
  dataset = dataset.map(utils.read_tfrecord_builder(is_training=True, seq_length=128, regression=False))
  dataset = dataset.flat_map(
              lambda x: tf.data.Dataset.from_tensors(x).repeat(utils.oversample_classes(x, 'founta'))
          )

  iterator = dataset.make_one_shot_iterator()
  next_element = iterator.get_next()

  count = [0,0,0,0]
  with tf.compat.v1.Session() as sess:
    try:
      while True:
        e = sess.run(next_element)
        count[e['label_ids']] += 1
    except tf.errors.OutOfRangeError:
      pass

count = np.asarray(count)

print(count)
print(count/count.sum())