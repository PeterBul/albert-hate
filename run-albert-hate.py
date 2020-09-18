from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow.compat.v1 as tf
import sys
import os
import argparse
import datetime
import pickle
import json
import six
import time
import copy


model_pickle_filepath = "master-model-version.pickle"

if not os.path.exists(model_pickle_filepath):
    model_version = 0
else:
    with open(model_pickle_filepath, 'rb') as handle:
        model_version = pickle.load(handle)
    model_version += 1

with open(model_pickle_filepath, 'wb') as handle:
    pickle.dump(model_version, handle)

parser = argparse.ArgumentParser()

parser.add_argument('--model-dir', dest='model_dir', type=str, default='models' + os.sep + 'albert-model', help='The name of the folder to save the model')
parser.add_argument('--ds', dest='dataset', type=str, default='solid', help='The name of the dataset to be used. For now,  davidson or solid')
parser.add_argument('--bs', dest='batch_size', type=int, default=32, help='Batch size')
parser.add_argument('--ll', dest='linear_layers', type=int, default=0, help="Number of linear layers to add on top of ALBERT")
parser.add_argument('--it', dest='iterations', type=int, default=1000, help='Number of iterations to train for. If epochs is set, this is overridden.')
parser.add_argument('--ws', dest='warmup_steps', type=int, default=0, help='Number of warmupsteps to be used.')

parser.add_argument('--mod', dest='model_size', type=str, default='base', help='The model size of albert. For now, "base" and "large" is supported.')
parser.add_argument('--seq-len', dest='sequence_length', type=int, default=128, help='The max sequence length of the model. This requires a dataset that is modeled for this.')
parser.add_argument('--epochs', dest='epochs', type=int, default=-1, help='Number of epochs to train. If not set, iterations are used instead.')
parser.add_argument('--ds-len', dest='dataset_length', type=int, default=-1, help='Length of dataset. Is needed to caluclate iterations if epochs is used.')
parser.add_argument('--opt', dest='optimizer', type=str, default='adamw', help='The optimizer to use. Supported optimizers are adam, adamw and lamb')
parser.add_argument('--lr', dest='learning_rate', type=float, default=1e-5, help='Learning rate')
parser.add_argument('--cp', dest='init_checkpoint', type=str, default=None, help="Checkpoint to train from")
parser.add_argument('--not-prob', dest='negative_class_prob', default=-1.0, type=float, help='Fraction of dataset containing negative labels')
parser.add_argument('--reg', dest='regression', default=False, action='store_true', help='True if using regression data for training')
parser.add_argument('--test', dest='test', default=False, action='store_true', help='Set flag to test')


args = parser.parse_args()


target_names = {'davidson': ['hateful', 'offensive', 'neither'], 'olid': ['NOT', 'OFF'], 'solid': ['NOT', 'OFF']}

BATCH_SIZE = args.batch_size

using_resampling = args.negative_class_prob != -1

using_epochs = args.epochs != -1 and args.dataset_length != -1

assert args.epochs == -1 or using_epochs , "If epochs argument is set, epochs are used. Make sure to set both epochs and ds-length if using epochs"

if using_epochs:
    ITERATIONS = int((args.epochs * args.dataset_length) / BATCH_SIZE)
else:
    ITERATIONS = args.iterations

# number of times we repeat the dataset is: epochs or iterations * batch_size / dataset_length
# TODO: find a better way of retrieving dataset length
if args.epochs == -1:
    epochs = int(ITERATIONS * BATCH_SIZE / args.dataset_length)
else:
    epochs = args.epochs



DIR = os.path.dirname(__file__)
if not args.test:
    MODEL_DIR = "{}-{}-{}-{}-{}-{}-{}".format(args.model_dir, datetime.date.today(), str(model_version), args.model_size, str(ITERATIONS), str(epochs), str(BATCH_SIZE))
else:
    MODEL_DIR = args.model_dir
PATH_DATASET= 'data' + os.sep + 'tfrecords'
FILE_TRAIN = PATH_DATASET + os.sep + args.dataset + os.sep + 'train-' + str(args.sequence_length) + '.tfrecords'
FILE_DEV = PATH_DATASET + os.sep + args.dataset + os.sep + 'dev-' + str(args.sequence_length) + '.tfrecords'
FILE_TEST = PATH_DATASET + os.sep + args.dataset + os.sep + 'test' + os.sep + 'test-' + str(args.sequence_length) + '.tfrecords'
#FILE_TRAIN = PATH_DATASET + os.sep + 'olid-2020-full-'+ ("reg-" if args.regression else "") + str(args.sequence_length) + '.tfrecords'
#FILE_DEV = PATH_DATASET + os.sep + 'olid-2019-full-' + str(args.sequence_length) + '.tfrecords'
#FILE_TEST = PATH_DATASET + os.sep + 'test-2020-' + str(args.sequence_length) + '.tfrecords'
ALBERT_PRETRAINED_PATH = 'albert_' + args.model_size
SST_2_WS = 1256

ALBERT_PATH = './ALBERT-master'
sys.path.append(ALBERT_PATH)

import optimization                                                     # pylint: disable=import-error
import modeling                                                         # pylint: disable=import-error
import classifier_utils                                                 # pylint: disable=import-error

tf.logging.set_verbosity(tf.logging.INFO)

tf.logging.info('Model directory: ' + MODEL_DIR)

tf.logging.info("------------ Arguments --------------")
for arg in vars(args):
    tf.logging.info(' {} {}'.format(arg, getattr(args, arg) or ''))
tf.logging.info("-------------------------------------")

tf.logging.info('Epochs: \t' + str(epochs))
tf.logging.info('Iterations: \t' +  str(ITERATIONS))


def read_tfrecord_builder(is_training, seq_length=args.sequence_length, regression=args.regression):
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

def train_input_fn(batch_size):
    ds = tf.data.TFRecordDataset(FILE_TRAIN)
    
    ds = ds.map(read_tfrecord_builder(is_training=True))
    if using_resampling:
        tf.logging.info("Resampling dataset with oversampling and undersampling")
        ds = ds.flat_map(
            lambda x: tf.data.Dataset.from_tensors(x).repeat(oversample_classes(x))
        )
        ds = ds.filter(undersampling_filter)
    ds = ds.shuffle(2048).repeat().batch(batch_size, drop_remainder=False)
    return ds

def eval_input_fn(batch_size, test=False):
    if not test:
        tf.logging.info('Using DEV dataset: {}'.format(FILE_DEV))
        ds = tf.data.TFRecordDataset(FILE_DEV)
    else:
        tf.logging.info('Using TEST dataset: {}'.format(FILE_TEST))
        ds = tf.data.TFRecordDataset(FILE_TEST)
    assert batch_size is not None, "batch_size must not be None"
    ds = ds.map(read_tfrecord_builder(is_training=False)).batch(batch_size)
    return ds


def model_fn_builder(regression=args.regression):    
    def my_model_fn(features, labels, mode):

        albert_config = modeling.AlbertConfig.from_json_file(ALBERT_PRETRAINED_PATH + os.sep + 'albert_config.json')

        tvars = tf.trainable_variables()
        initialized_variable_names = {}

        init_checkpoint = ALBERT_PRETRAINED_PATH + os.sep + 'model.ckpt-best'
        (assignment_map, initialized_variable_names) = \
            modeling.get_assignment_map_from_checkpoint(tvars, init_checkpoint)
        tf.train.init_from_checkpoint(init_checkpoint, assignment_map)
        tf.logging.info("**** Trainable Variables ****")
        for var in tvars:
            init_string = ""
            if var.name in initialized_variable_names:
                init_string = ", *INIT_FROM_CKPT*"
            tf.logging.info("  name = %s, shape = %s%s", var.name, var.shape, init_string)

        
        if mode == tf.estimator.ModeKeys.PREDICT:
            tf.logging.info("my_model_fn: PREDICT, {}".format(mode))
        elif mode == tf.estimator.ModeKeys.EVAL:
            tf.logging.info("my_model_fn: EVAL, {}".format(mode))
        elif mode == tf.estimator.ModeKeys.TRAIN:
            tf.logging.info("my_model_fn: TRAIN, {}".format(mode))

        
        input_ids = features['input_ids']
        input_mask = features['input_mask']
        segment_ids = features['segment_ids']
        

        is_training = mode == tf.estimator.ModeKeys.TRAIN

        if regression and is_training:
            label_ids = features['average']
        else:
            label_ids = features['label_ids']

        (loss, per_example_loss, probabilities, logits, predictions) = \
            classifier_utils.create_model(albert_config, is_training, input_ids, input_mask, \
                segment_ids, label_ids, 3, False, 'hsc', None, args.linear_layers, regression)
        
        
        
        
        if mode == tf.estimator.ModeKeys.PREDICT:
            predictions={
                "probabilities": probabilities,
                "predictions": predictions,
                "logits": logits
            }
            return tf.estimator.EstimatorSpec(mode, predictions=predictions)

        predictions = tf.argmax(logits, axis=-1, output_type=tf.int32)
        
        # NOT offensive has label 0, OFF (offensive) has label 1. OFF is thus positive label

        accuracy = tf.metrics.accuracy(labels=label_ids, predictions=predictions, name='acc_op')
        auc = tf.metrics.auc(label_ids, predictions=predictions, name='auc_op')
        eval_loss = tf.metrics.mean(values=per_example_loss)

        metrics = get_metrics(label_ids, predictions, target_names=target_names[args.dataset])
        metrics['accuracy'] = accuracy
        metrics['auc'] = auc
        metrics['eval_loss'] = eval_loss

        for k, v in metrics.items():
            tf.summary.scalar(k, v[1]) 

        tf.summary.scalar('loss', loss)

        if mode == tf.estimator.ModeKeys.EVAL:
            return tf.estimator.EstimatorSpec(mode, loss=loss, eval_metric_ops=metrics)
        
        assert mode == tf.estimator.ModeKeys.TRAIN, "TRAIN is only ModeKey left"

        if args.optimizer == 'adam':
            optimizer = tf.train.AdamOptimizer(args.learning_rate)
            train_op = optimizer.minimize(loss, global_step=tf.train.get_global_step())
        elif args.optimizer == 'adamw' or args.optimizer == 'lamb':
            train_op = optimization.create_optimizer(
            loss, args.learning_rate, ITERATIONS, args.warmup_steps,
            False, args.optimizer)
        else:
            raise ValueError('Optimizer has to be "adam", "adamw" or "lamb"')
        
        if args.optimizer == 'adam':
            grads = optimizer.compute_gradients(loss)
            for grad, var in grads:
                if grad is not None:
                    tf.summary.histogram(var.op.name + '/gradients', grad)
        for var in tf.trainable_variables():
            tf.summary.histogram(var.name, var)
        return tf.estimator.EstimatorSpec(mode, loss=loss, train_op=train_op)
    return my_model_fn


def get_metrics(y_true, y_pred, target_names=['hateful', 'offensive', 'neither']):

    assert y_true.shape[-1] == len(target_names), "Number of target names, {}, must match the number of classes: {}".format(len(target_names), y_true.shape[-1])

    target_names = [tn.lower() for tn in target_names]
    
    metrics = {}

    precisions = [0,0,0]
    recalls = [0,0,0]
    f1s = [0, 0, 0]

    y_true = tf.cast(y_true, tf.float64)
    y_pred = tf.cast(y_pred, tf.float64)
    
    support = tf.count_nonzero(y_true, axis=0)
    # axis=None -> micro, axis=1 -> macro
    for i, axis in enumerate([None, 0]):
        TP = tf.count_nonzero(y_pred * y_true, axis=axis, dtype=tf.dtypes.float64)
        FP = tf.count_nonzero(y_pred * (y_true - 1), axis=axis, dtype=tf.dtypes.float64)
        FN = tf.count_nonzero((y_pred - 1) * y_true, axis=axis, dtype=tf.dtypes.float64)

        precision = tf.math.divide_no_nan(TP, (TP + FP))
        recall = tf.math.divide_no_nan(TP, (TP + FN))
        f1 = tf.math.divide_no_nan(2 * precision * recall, (precision + recall))

        precisions[i] = tf.reduce_mean(precision)
        recalls[i] = tf.reduce_mean(recall)
        f1s[i] = tf.reduce_mean(f1)

    weights = tf.reduce_sum(y_true, axis=0)
    weights /= tf.reduce_sum(weights)
    
    precisions[2] = tf.reduce_sum(precision * weights)
    recalls[2] = tf.reduce_sum(recall * weights)
    f1s[2] = tf.reduce_sum(f1 * weights)

    tot_supp = tf.reduce_sum(support)
    supports = [tot_supp for i in range(3)]

    for i, metric in enumerate(['precision', 'recall', 'f1', 'support']):
        for j, sublist in enumerate([target_names, ['micro-avg', 'macro-avg', 'weighted-avg']]):
            for k, label in enumerate(sublist):
                key = '{}_{}'.format(metric, label)
                lookup = [[precision, precisions], [recall, recalls], [f1, f1s], [support, supports]]
                metrics[key] = lookup[i][j][k]
    
    metrics = {k: (v, tf.identity(v)) for k, v in metrics}
 
    return metrics


# sampling parameters use it wisely 
oversampling_coef = 0.9 # if equal to 0 then oversample_classes() always returns 1
undersampling_coef = 0.9 # if equal to 0 then undersampling_filter() always returns True

def oversample_classes(example):
    """
    Returns the number of copies of given example
    """
    label_id = example['label_ids']
    def f1(): return tf.constant(args.negative_class_prob)
    def f2(): return tf.constant(1 - args.negative_class_prob)
    class_prob = tf.cond(tf.math.equal(label_id, tf.constant(0)), f1, f2)
    class_target_prob = tf.constant(0.5)
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

    return repeat_count + residual_acceptance


def undersampling_filter(example):
    """
    Computes if given example is rejected or not.
    """
    label_id = example['label_ids']
    def f1(): return tf.constant(args.negative_class_prob)
    def f2(): return tf.constant(1 - args.negative_class_prob)
    class_prob = tf.cond(tf.math.equal(label_id, tf.constant(0)), f1, f2)
    class_target_prob = tf.constant(0.5)
    prob_ratio = tf.cast(class_target_prob/class_prob, dtype=tf.float32)
    prob_ratio = prob_ratio ** undersampling_coef
    prob_ratio = tf.minimum(prob_ratio, 1.0)

    acceptance = tf.less_equal(tf.random_uniform([], dtype=tf.float32), prob_ratio)
    # predicate must return a scalar boolean tensor
    return acceptance

if __name__ == "__main__":
    tf.debugging.set_log_device_placement(True)
    strategy = tf.distribute.MirroredStrategy()
    config = tf.estimator.RunConfig(train_distribute=strategy)

    classifier = tf.estimator.Estimator(
        model_fn=model_fn_builder(),
        model_dir=MODEL_DIR,
        config=config
        )

    if args.test:
        classifier.evaluate(input_fn=lambda:eval_input_fn(2, test=True), steps=None)
    else:
        early_stopping = tf.estimator.experimental.stop_if_no_decrease_hook(classifier, metric_name="loss", max_steps_without_decrease=10000, min_steps=100)
        train_spec = tf.estimator.TrainSpec(input_fn=lambda: train_input_fn(BATCH_SIZE), max_steps=ITERATIONS, hooks=[early_stopping])
        eval_spec = tf.estimator.EvalSpec(input_fn=lambda:eval_input_fn(BATCH_SIZE), steps=None)

        tf.estimator.train_and_evaluate(classifier, train_spec, eval_spec)
    
