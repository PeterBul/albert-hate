from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys
import os
dirname = os.path.dirname(__file__)
albert_path = os.path.join(dirname, 'ALBERT-master')
sys.path.append(albert_path)

import tensorflow.compat.v1 as tf                                       # pylint: disable=import-error

from tensorflow.contrib import data as contrib_data                     # pylint: disable=import-error
from tensorflow.contrib import metrics as contrib_metrics               # pylint: disable=import-error

import optimization                                                     # pylint: disable=import-error
import modeling                                                         # pylint: disable=import-error
import classifier_utils                                                 # pylint: disable=import-error

import json
import six
import time
import copy

ITERATIONS = 20935
BATCH_SIZE = 64

def read_tfrecord(serialized_example, seq_length=512):
    """
    Map output of TFRecord which only has 1-D tensors to multidimensional tensors, and give input from dataset in proper format.
    Each example comes with several possible captions, so choose one randomly. This can be changed to using a non-probabilistic method.
    """
    feature_description = {
        "input_ids": tf.FixedLenFeature([seq_length], tf.int64),
        "input_mask": tf.FixedLenFeature([seq_length], tf.int64),
        "segment_ids": tf.FixedLenFeature([seq_length], tf.int64),
        "label_ids": tf.FixedLenFeature([], tf.int64),
        "is_real_example": tf.FixedLenFeature([], tf.int64),
    }
    
    example = decode_record(serialized_example, feature_description)
    
    return example

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

def fully_connected(input_tensor, input_size, hidden_size, name="fc", bias=True, activation=tf.nn.relu):
    W = tf.get_variable(name + "_W", [input_size, hidden_size], initializer=tf.truncated_normal_initializer())
    b = tf.get_variable(name + "_b", [hidden_size], initializer=tf.constant_initializer(0.0))
    if activation is not None:
        return activation(tf.nn.xw_plus_b(input_tensor, W, b))
    else:
        return tf.nn.xw_plus_b(input_tensor, W, b)

if __name__ == "__main__":
    tf.reset_default_graph()
    albert_config = modeling.AlbertConfig.from_json_file(os.path.join(dirname, 'albert_base/albert_config.json'))
    
    ds = tf.data.TFRecordDataset(os.path.join(dirname, './data/tfrecords/train-olid.tfrecords'))
    ds = ds.map(read_tfrecord)
    ds = ds.repeat()
    ds = ds.shuffle(buffer_size=100)
    ds = ds.batch(batch_size=BATCH_SIZE, drop_remainder=False)
    iterator = ds.make_one_shot_iterator()
    X = iterator.get_next()
    (loss, per_example_loss, probabilities, logits, predictions) = \
        classifier_utils.create_model(albert_config, True, X['input_ids'], X['input_mask'], \
            X['segment_ids'], X['label_ids'], 2, False, 'hsc', None)

    tvars = tf.trainable_variables()
    initialized_variable_names = {}
    
    init_checkpoint = os.path.join(dirname, 'albert_base/model.ckpt-best')
    (assignment_map, initialized_variable_names) = \
        modeling.get_assignment_map_from_checkpoint(tvars, init_checkpoint)
    tf.train.init_from_checkpoint(init_checkpoint, assignment_map)
    tf.logging.info("**** Trainable Variables ****")
    for var in tvars:
        init_string = ""
        if var.name in initialized_variable_names:
            init_string = ", *INIT_FROM_CKPT*"
        tf.logging.info("  name = %s, shape = %s%s", var.name, var.shape, init_string)
    


    
    #optimizer = optimization.create_optimizer(loss, 1e-5, 20935, 1256, False, optimizer="adamw")
    optimizer = tf.train.AdamOptimizer(1e-5).minimize(loss)
    loss_summary = tf.summary.scalar(name='Loss', tensor=loss)
    
    init_op = tf.global_variables_initializer()
    saver = tf.train.Saver()

    with tf.Session(config=tf.ConfigProto(
        allow_soft_placement=True)) as sess:
        
        writer = tf.summary.FileWriter(os.path.join(os.path.join(dirname, 'graphs'), str(time.time_ns())), sess.graph)
        sess.run(init_op)
        try:
            for i in range(ITERATIONS):
                _, loss_val, loss_summary_val = sess.run([optimizer, loss, loss_summary])
                writer.add_summary(loss_summary_val, i)
                if i % 100 == 0:
                    print("Loss: " + str(loss_val))
                    save_path = saver.save(sess, "checkpoints/model.ckpt")
                    print("Model saved in path: %s" % save_path)
        except tf.errors.OutOfRangeError:
            pass
