from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys
sys.path.append('./ALBERT-master')

import mpi4py as MPI
import tensorflow.compat.v1 as tf                                       # pylint: disable=import-error
import tensorflow_hub as hub

import optimization                                                     # pylint: disable=import-error

import numpy
import json
import six
import datetime
import copy




ITERATIONS = 20935
BATCH_SIZE = 16

def read_tfrecord(serialized_example, seq_length=512):
    """
    Map output of TFRecord which only has 1-D tensors to multidimensional tensors, and give input from dataset in proper format.
    Each example comes with several possible captions, so choose one randomly. This can be changed to using a non-probabilistic method.
    """
    feature_description = {
        "input_ids": tf.io.FixedLenFeature([seq_length], tf.int64),
        "input_mask": tf.io.FixedLenFeature([seq_length], tf.int64),
        "segment_ids": tf.io.FixedLenFeature([seq_length], tf.int64),
        "label_ids": tf.io.FixedLenFeature([], tf.int64),
        "is_real_example": tf.io.FixedLenFeature([], tf.int64),
    }
    
    example = decode_record(serialized_example, feature_description)
    
    return example

def decode_record(record, feature_description):
    """Decodes a record to a TensorFlow example."""
    example = tf.io.parse_single_example(record, feature_description)

    # tf.Example only supports tf.int64, but the TPU only supports tf.int32.
    # So cast all int64 to int32.
    for name in list(example.keys()):
      t = example[name]
      if t.dtype == tf.int64:
        t = tf.cast(t, dtype=tf.int32)
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
    tags = set()
    tags.add("train")
    albert_module = hub.Module(
        "https://tfhub.dev/google/albert_base/3", tags=tags,
        trainable=True)
    
    ds = tf.data.TFRecordDataset('./data/tfrecords/train-olid.tfrecords')
    ds = ds.map(read_tfrecord)
    ds = ds.repeat()
    ds = ds.shuffle(buffer_size=100)
    ds = ds.batch(batch_size=BATCH_SIZE, drop_remainder=False)
    iterator = ds.make_one_shot_iterator()
    X = iterator.get_next()

    albert_inputs = dict(
      input_ids=X['input_ids'],
      input_mask=X['input_mask'],
      segment_ids=X['segment_ids'])
    albert_outputs = albert_module(albert_inputs, signature="tokens", as_dict=True)
    pooled_output = albert_outputs["pooled_output"]
    hidden_size = pooled_output.shape[-1].value
    output_weights = tf.get_variable(
        "output_weights", [hidden_size, 2],
        initializer=tf.truncated_normal_initializer(stddev=0.02))
    output_bias = tf.get_variable(
        "output_bias", [2], initializer=tf.zeros_initializer())
    output_layer = tf.nn.dropout(pooled_output, rate=0.1)
    logits = tf.matmul(output_layer, output_weights)
    logits = tf.nn.bias_add(logits, output_bias)
    
    
    probabilities = tf.nn.softmax(logits, axis=-1)
    predictions = tf.argmax(probabilities, axis=-1, output_type=tf.int32)
    log_probs = tf.nn.log_softmax(logits, axis=-1)

    one_hot_labels = tf.one_hot(X['label_ids'], depth=2, dtype=tf.float32)
    per_example_loss = -tf.reduce_sum(one_hot_labels * log_probs, axis=-1)
    loss = tf.reduce_mean(per_example_loss)
    optimizer = optimization.create_optimizer(loss, 1e-5, 20935, 1256, False, optimizer="adamw")

    loss_summary = tf.summary.scalar(name='Loss', tensor=loss)

    with tf.Session() as sess:
        if tf.device("/gpu:0"):
            print("GPU implemented")
        else:
            print("GPU not implemented")
        if tf.test.is_gpu_available():
            print("GPU is available")
        else:
            print("GPU is unavailable")
        writer = tf.summary.FileWriter('./graphs/' + datetime.datetime.now().strftime("%Y-%m-%d--%H-%M"), sess.graph)
        sess.run(tf.global_variables_initializer())
        try:
            for i in range(ITERATIONS):
                _, loss_val, loss_summary_val = sess.run([optimizer, loss, loss_summary])
                writer.add_summary(loss_summary_val, i)
        except tf.errors.OutOfRangeError:
            pass

