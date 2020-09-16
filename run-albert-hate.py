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

model_pickle_filepath = "model-version.pickle"

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
parser.add_argument('--bs', dest='batch_size', type=int, default=32, help='Batch size')
parser.add_argument('--it', dest='iterations', type=int, default=1000, help='Number of iterations to train for. If epochs is set, this is overridden.')
parser.add_argument('--mod', dest='model_size', type=str, default='base', help='The model size of albert. For now, "base" and "large" is supported.')
parser.add_argument('--seq-len', dest='sequence_length', type=int, default=512, help='The max sequence length of the model. This requires a dataset that is modeled for this.')
parser.add_argument('--epochs', dest='epochs', type=int, default=-1, help='Number of epochs to train. If not set, iterations are used instead.')
parser.add_argument('--ds-len', dest='dataset_length', type=int, default=-1, help='Length of dataset. Is needed to caluclate iterations if epochs is used.')
parser.add_argument('--opt', dest='optimizer', type=str, default='adam', help='The optimizer to use. Supported optimizers are adam, adamw and lamb')
parser.add_argument('--ws', dest='warmup_steps', type=int, default=0, help='Number of warmupsteps to be used.')
parser.add_argument('--lr', dest='learning_rate', type=float, default=1e-5, help='Learning rate')
parser.add_argument('--cp', dest='init_checkpoint', type=str, default=None, help="Checkpoint to train from")
parser.add_argument('--not-prob', dest='negative_class_prob', default=-1.0, type=float, help='Fraction of dataset containing negative labels')
parser.add_argument('--reg', dest='regression', default=False, action='store_true', help='True if using regression data for training')
parser.add_argument('--test', dest='test', default=False, action='store_true', help='Set flag to test')



args = parser.parse_args()


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
FILE_TRAIN = PATH_DATASET + os.sep + 'olid-2020-full-'+ ("reg-" if args.regression else "") + str(args.sequence_length) + '.tfrecords'
FILE_DEV = PATH_DATASET + os.sep + 'olid-2019-full-' + str(args.sequence_length) + '.tfrecords'
FILE_TEST = PATH_DATASET + os.sep + 'test-2020-' + str(args.sequence_length) + '.tfrecords'
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
                segment_ids, label_ids, 2, False, 'hsc', None, regression)
        
        
        
        
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
        prec, prec_op = tf.metrics.precision(label_ids, predictions=predictions, name='precision_op')
        rec, rec_op = tf.metrics.recall(label_ids, predictions=predictions, name='recall_op')
        f1 = 2 * (rec * prec)/(rec + prec)
        f1_op = tf.identity(f1)
        precision = (prec, prec_op)
        recall = (rec, rec_op)
        f1 = (f1, f1_op)
        
        auc = tf.metrics.auc(label_ids, predictions=predictions, name='auc_op')
        eval_loss = tf.metrics.mean(values=per_example_loss)

        # Own calculations to double check

        false_pos = tf.metrics.false_positives(labels=label_ids, predictions=predictions, name='false_positives_op')
        false_neg = tf.metrics.false_negatives(labels=label_ids, predictions=predictions, name='false_negatives_op')
        true_pos = tf.metrics.true_positives(labels=label_ids, predictions=predictions, name='true_positives_op')
        true_neg = tf.metrics.true_negatives(labels=label_ids, predictions=predictions, name='true_negatives_op')

        fp, _ = false_pos
        fn, _ = false_neg
        tp, _ = true_pos
        tn, _ = true_neg

        # Precision OFF (precision) = tp/(tp + fp)
        prec_off = tp/(tp + fp)
        prec_off_op = tf.identity(prec_off)
        precision_off_metric = (prec_off, prec_off_op)

        # Recall OFF (recall) = tp/(tp + fn)
        rec_off = tp/(tp + fn)
        rec_off_op = tf.identity(rec_off)
        recall_off_metric = (rec_off, rec_off_op)

        # Precision NOT (negative predictive value) = tn/(tn + fn)
        prec_not = tn/(tn + fn)
        prec_not_op = tf.identity(prec_not)
        precision_not_metric = (prec_not, prec_not_op)

        # Recall NOT (True negative rate / specificity) = tn/(tn + fp)
        rec_not = tn/(tn + fp)
        rec_not_op = tf.identity(rec_not)
        recall_not_metric = (rec_not, rec_not_op)

        f1_off = 2 * (rec_off * prec_off)/(rec_off + prec_off)
        f1_off_op = tf.identity(f1_off)
        f1_offensive_metric = (f1_off, f1_off_op)

        f1_not = 2 * (rec_not * prec_not)/(rec_not + prec_not)
        f1_not_op = tf.identity(f1_not)
        f1_not_metric = (f1_not, f1_not_op)






        metrics = {
            'eval_accuracy': accuracy, 
            'eval_precision': precision, 
            'eval_recall': recall,
            'eval_precision_offensive': precision_off_metric,
            'eval_recall_offensive': recall_off_metric,
            'eval_precision_not': precision_not_metric,
            'eval_recall_not': recall_not_metric, 
            'eval_auc': auc, 
            'eval_loss': eval_loss, 
            'f1': f1,
            'f1_offensive': f1_offensive_metric,
            'f1_not': f1_not_metric, 
            'true_positives': true_pos,
            'true_negatives': true_neg,
            'false_positives': false_pos,
            'false_negatives': false_neg
            }
        tf.summary.scalar('accuracy', accuracy[1])
        tf.summary.scalar('precision', precision[1])
        tf.summary.scalar('recall', recall[1])
        tf.summary.scalar('auc', auc[1])
        tf.summary.scalar('mean_per_example_loss', eval_loss[1])
        tf.summary.scalar('loss', loss)
        tf.summary.scalar('true_positives', true_pos[1])
        tf.summary.scalar('true_negatives', true_neg[1])
        tf.summary.scalar('false_positives', false_pos[1])
        tf.summary.scalar('false_negatives', false_neg[1])
        tf.summary.scalar('f1', f1[1])

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
    #tf.debugging.set_log_device_placement(True)
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
    
