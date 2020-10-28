from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import sys
import os
import argparse
import datetime
import pickle
import json
import numpy as np
import pandas as pd
import six
import time
import copy
import wandb
import sentencepiece as spm
import re
import utils
import metrics
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from constants import target_names, class_probabilities, num_labels

parser = argparse.ArgumentParser()


# Parameters
parser.add_argument('--albert_dropout', type=float, default=0)
parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
parser.add_argument('--iterations', type=int, default=1000, help='Number of iterations to train for. If epochs is set, this is overridden.')
parser.add_argument('--linear_layers', type=int, default=0, help="Number of linear layers to add on top of ALBERT")
parser.add_argument('--learning_rate', type=float, default=1e-5, help='Learning rate')
parser.add_argument('--model_size', type=str, default='base', help='The model size of albert. For now, "base" and "large" is supported.')
parser.add_argument('--optimizer', type=str, default='adamw', help='The optimizer to use. Supported optimizers are adam, adamw and lamb')
parser.add_argument('--use_seq_out', type=utils.str2bool, const=True, default=False, nargs='?', help='Set flag to use sequence output instead of first token')
parser.add_argument('--warmup_steps', type=int, default=0, help='Number of warmupsteps to be used.')

parser.add_argument('--sequence_length', type=int, default=128, help='The max sequence length of the model. This requires a dataset that is modeled for this.')
parser.add_argument('--regression', type=utils.str2bool, const=True, default=False, nargs='?', help='True if using regression data for training')
parser.add_argument('--accumulation_steps', type=int, default=1, help='Set number of steps to accumulate for before doing update to parameters')
parser.add_argument('--no_resampling', dest='use_resampling', default=True, action='store_false')
parser.add_argument('--num_labels', default=None)

# Control arguments
parser.add_argument('--config_path', type=str, default=None, help="Path to model config for albert hate")
parser.add_argument('--init_checkpoint', type=str, default=None, help="Checkpoint to train from")
parser.add_argument('--wandb_run_path', type=str, default=None)
parser.add_argument('--model_dir', type=str, help='The name of the folder to save the model')
parser.add_argument('--predict', type=utils.str2bool, const=True, default=False, nargs='?')
parser.add_argument('--debug', type=utils.str2bool, const=True, default=False, nargs='?', help='Set flag to debug')
parser.add_argument('--test', type=utils.str2bool, const=True, default=False, nargs='?', help='Set flag to test')
parser.add_argument('--dataset', type=str, default='solid', help='The name of the dataset to be used. For now,  davidson or solid')
parser.add_argument('--task', type=str, default='a', help='Task a, b or c for solid/olid dataset')
parser.add_argument('--tree_predict', type=utils.str2bool, const=True, default=False, nargs='?')
parser.add_argument('--ensamble', type=utils.str2bool, const=True, default=False, nargs='?')

# Arguments that aren't very important
parser.add_argument('--epochs', type=int, default=-1, help='Number of epochs to train. If not set, iterations are used instead.')
parser.add_argument('--dataset_length', type=int, default=-1, help='Length of dataset. Is needed to caluclate iterations if epochs is used.')

args = parser.parse_args()

use_accumulation = args.accumulation_steps > 1

class AlbertHateConfig(object):
    def __init__(self,
                    linear_layers=0,
                    model_dir=None,
                    model_size='base',
                    num_labels=num_labels[args.dataset],
                    regression=False,
                    sequence_length=128,
                    use_seq_out=True,
                    best_checkpoint=None
                    ):
        self.linear_layers = linear_layers
        self.model_dir = model_dir
        self.model_size = model_size
        self.num_labels = num_labels
        self.regression = utils.str2bool(regression)
        self.sequence_length = sequence_length
        self.use_seq_out = utils.str2bool(use_seq_out)
        self.best_checkpoint = best_checkpoint

    
    @classmethod
    def from_dict(cls, json_object):
        """Constructs a `AlbertHateConfig` from a Python dictionary of parameters."""
        config = AlbertHateConfig()
        for (key, value) in six.iteritems(json_object):
            #if isinstance(config.__dict__[key], bool):
            #    value = utils.str2bool(value)
            config.__dict__[key] = value
        return config

    @classmethod
    def from_json_file(cls, json_file):
        """Constructs a `AlbertHateConfig` from a json file of parameters."""
        with tf.gfile.GFile(json_file, "r") as reader:
            text = reader.read()
        return cls.from_dict(json.loads(text))

    def to_dict(self):
        """Serializes this instance to a Python dictionary."""
        output = copy.deepcopy(self.__dict__)
        return output

    def to_json_string(self):
        """Serializes this instance to a JSON string."""
        return json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n"

print(args.__dict__)

if args.config_path:
    model_config = AlbertHateConfig.from_json_file(args.config_path)
    for key, var in six.iteritems(model_config.__dict__):
        args.__dict__[key] = var

wandb.init(
  project="albert-hate",
  config=args.__dict__,
  sync_tensorboard=True
)

model_dir = wandb.run.dir if args.model_dir is None else args.model_dir

config = wandb.config





using_epochs = args.epochs != -1 and args.dataset_length != -1
assert args.epochs == -1 or using_epochs , "If epochs argument is set, epochs are used. Make sure to set both epochs and ds-length if using epochs"
assert args.task in ('a', 'b', 'c'), "Task has to be a, b, or c if specifying task. Task is only relevant for solid/olid dataset"

if using_epochs:
    ITERATIONS = int((args.epochs * args.dataset_length) / args.batch_size)
else:
    ITERATIONS = args.iterations

if args.epochs == -1:
    epochs = int(ITERATIONS * args.batch_size / args.dataset_length)
else:
    epochs = args.epochs

DIR = os.path.dirname(__file__)
PATH_DATASET= 'data' + os.sep + 'tfrecords'

if args.dataset == 'davidson':
    FILE_TRAIN = PATH_DATASET + os.sep + args.dataset + os.sep + 'train-' + str(128) + '.tfrecords'
    FILE_DEV = PATH_DATASET + os.sep + args.dataset + os.sep + 'dev-' + str(128) + '.tfrecords'
    FILE_TEST = PATH_DATASET + os.sep + args.dataset + os.sep + 'test' + os.sep + 'test-' + str(128) + '.tfrecords'
elif args.dataset == 'solid':
    FILE_TRAIN = PATH_DATASET + os.sep + 'solid' + os.sep + 'task-' + args.task + os.sep + 'solid-' + str(args.sequence_length) + '.tfrecords'
    FILE_DEV = PATH_DATASET + os.sep + 'olid' + os.sep + 'task-' + args.task + os.sep + 'olid-' + str(args.sequence_length) + '.tfrecords'
    FILE_TEST = PATH_DATASET + os.sep + 'solid' + os.sep + 'task-' + args.task + os.sep + 'test' + os.sep + 'solid-' + str(args.sequence_length) + '.tfrecords'
elif args.dataset =='converted':
    FILE_TRAIN = PATH_DATASET + os.sep + 'converted' + os.sep + 'solid-' + str(args.sequence_length) + '.tfrecords'
    FILE_DEV = PATH_DATASET + os.sep + 'converted' + os.sep + 'olid-' + str(args.sequence_length) + '.tfrecords'
    FILE_TEST = PATH_DATASET + os.sep + 'converted' + os.sep + 'test-' + str(args.sequence_length) + '.tfrecords'
elif args.dataset == 'founta':
    FILE_TRAIN = PATH_DATASET + os.sep + args.dataset + os.sep + 'train-' + str(args.sequence_length) + '.tfrecords'
    FILE_DEV = PATH_DATASET + os.sep + args.dataset + os.sep + 'dev-' + str(args.sequence_length) + '.tfrecords'
    FILE_TEST = PATH_DATASET + os.sep + args.dataset + os.sep + 'test-' + str(args.sequence_length) + '.tfrecords'
else:
    #FILE_TRAIN = PATH_DATASET + os.sep + 'olid-2020-full-'+ ("reg-" if args.regression else "") + str(args.sequence_length) + '.tfrecords'
    #FILE_DEV = PATH_DATASET + os.sep + 'olid-2019-full-' + str(args.sequence_length) + '.tfrecords'
    #FILE_TEST = PATH_DATASET + os.sep + 'test-2020-' + str(args.sequence_length) + '.tfrecords'
    raise ValueError("Dataset not supported")

if args.dataset in ('solid', 'olid'):
    args.dataset += '_' + args.task

if args.num_labels is None:
    args.num_labels = num_labels[args.dataset]

ALBERT_PRETRAINED_PATH = 'albert_' + args.model_size

ALBERT_PATH = './ALBERT-master'
sys.path.append(ALBERT_PATH)

import optimization                                                     # pylint: disable=import-error
import modeling                                                         # pylint: disable=import-error
import classifier_utils                                                 # pylint: disable=import-error
from gradient_accumulation import GradientAccumulationHook

tf.logging.set_verbosity(tf.logging.INFO)

tf.logging.info('Model directory: ' + model_dir)

tf.logging.info("------------ Arguments --------------")
for arg in vars(args):
    tf.logging.info(' {} {}'.format(arg, getattr(args, arg) or ''))
tf.logging.info("-------------------------------------")

tf.logging.info('Epochs: \t' + str(epochs))
tf.logging.info('Iterations: \t' +  str(ITERATIONS))


def k_fold_train(ds, folds, evaluate_on):
    first = True
    for i in range(0, folds):
        if i != evaluate_on:
            if first:
                new_ds = ds.shard(folds,i)
                first = False
            else:
                new_ds.concatenate(ds.shard(folds, i))
    return new_ds

def k_fold_eval(ds, folds, evaluate_on):
    return ds.shard(folds, evaluate_on)


def train_input_fn(batch_size, folds=1, evaluate_on=-1):
    do_kfold = folds > 1 and evaluate_on > -1
    tf.logging.info('Using TRAIN dataset: {}'.format(FILE_TRAIN))
    ds = tf.data.TFRecordDataset(FILE_TRAIN)
    #ds_eval = tf.data.TFRecordDataset(FILE_DEV)
    #ds = ds.concatenate(ds_eval)
    
    if do_kfold:
        ds_eval = tf.data.TFRecordDataset(FILE_DEV)
        ds = ds.concatenate(ds_eval)
        ds = k_fold_train(ds, folds, evaluate_on)

    ds = ds.map(utils.read_tfrecord_builder(is_training=True, seq_length=args.sequence_length, regression=args.regression))
    if args.use_resampling:
        tf.logging.info("Resampling dataset with oversampling and undersampling")
        ds = ds.flat_map(
            lambda x: tf.data.Dataset.from_tensors(x).repeat(oversample_classes(x))
        )
        #ds = ds.filter(undersampling_filter)
    
    ds = ds.shuffle(2048).repeat().batch(batch_size, drop_remainder=False)
    return ds

def eval_input_fn(batch_size, test=False, folds=1, evaluate_on=-1):
    do_kfold = folds > 1 and evaluate_on > -1
    if do_kfold:
        tf.logging.info('Using TRAIN dataset with k-fold with train set: {} on fold: {}'.format(FILE_TRAIN, evaluate_on))
        ds = tf.data.TFRecordDataset(FILE_TRAIN)
        ds_eval = tf.data.TFRecordDataset(FILE_DEV)
        ds = ds.concatenate(ds_eval)
        ds = k_fold_eval(ds, folds, evaluate_on)
    elif not test:
        tf.logging.info('Using DEV dataset: {}'.format(FILE_DEV))
        ds = tf.data.TFRecordDataset(FILE_DEV)
    else:
        tf.logging.info('Using TEST dataset: {}'.format(FILE_TEST))
        ds = tf.data.TFRecordDataset(FILE_TEST)
    assert batch_size is not None, "batch_size must not be None"
    ds = ds.map(utils.read_tfrecord_builder(is_training=False, seq_length=args.sequence_length))
    ds = ds.batch(batch_size)
    return ds

def tf_count(t, val):
    elements_equal_to_value = tf.equal(t, val)
    as_ints = tf.cast(elements_equal_to_value, tf.int32)
    count = tf.reduce_sum(as_ints)
    return count

def model_fn_builder(config=None):    
    def my_model_fn(features, labels, mode):
        

        training_hooks = []

        params = args if config is None else config

        albert_pretrained_path = 'albert_' + params.model_size

        albert_config = modeling.AlbertConfig.from_json_file(albert_pretrained_path + os.sep + 'albert_config.json')
        if args.albert_dropout is not None:
            albert_config.hidden_dropout_prob = args.albert_dropout
        #albert_config.attention_probs_dropout_prob = args.dropout

        
        input_ids = features['input_ids']
        input_mask = features['input_mask']
        segment_ids = features['segment_ids']
        

        tf.logging.info(input_ids)
        is_training = mode == tf.estimator.ModeKeys.TRAIN

        if params.regression and is_training:
            label_ids = features['average']
        else:
            label_ids = features['label_ids']



        (loss, per_example_loss, probabilities, logits, predictions) = classifier_utils.create_model(
                                                                            albert_config=albert_config,
                                                                            is_training=is_training, 
                                                                            input_ids=input_ids, 
                                                                            input_mask=input_mask, 
                                                                            segment_ids=segment_ids, 
                                                                            labels=label_ids,
                                                                            num_labels=params.num_labels,
                                                                            use_one_hot_embeddings=False,
                                                                            task_name='hsc', 
                                                                            hub_module=None, 
                                                                            linear_layers=params.linear_layers,
                                                                            regression=params.regression,
                                                                            use_sequence_output=params.use_seq_out)
        
        

        tvars = tf.trainable_variables()
        initialized_variable_names = {}
        if config and config.best_checkpoint:
            init_checkpoint = config.best_checkpoint
        else:
            init_checkpoint = albert_pretrained_path + os.sep + 'model.ckpt-best'
        (assignment_map, initialized_variable_names) = \
            modeling.get_assignment_map_from_checkpoint(tvars, init_checkpoint)

        tf.logging.info("Initialized variables:")
        for var in initialized_variable_names:
            tf.logging.info(var)
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
        
        
        if mode == tf.estimator.ModeKeys.PREDICT:
            predictions={
                "input_ids": input_ids,
                "gold": label_ids,
                "probabilities": probabilities,
                "predictions": predictions,
                "logits": logits
            }
            return tf.estimator.EstimatorSpec(mode, predictions=predictions)

        predictions = tf.argmax(logits, axis=-1, output_type=tf.int32)
        
        # NOT offensive has label 0, OFF (offensive) has label 1. OFF is thus positive label

        accuracy = tf.metrics.accuracy(labels=label_ids, predictions=predictions, name='acc_op')
        eval_loss = tf.metrics.mean(values=per_example_loss)

        metrics = get_metrics(label_ids, predictions)
        metrics['accuracy'] = accuracy
        
        global_step = tf.train.get_global_step()

        if mode == tf.estimator.ModeKeys.EVAL:
            metrics = {'eval_' + k: v for k, v in metrics.items()}
            metrics['eval_loss'] = eval_loss
            for k, v in metrics.items():
                tf.summary.scalar(k, v[1]) 
            return tf.estimator.EstimatorSpec(mode, loss=loss, eval_metric_ops=metrics)
        
        assert mode == tf.estimator.ModeKeys.TRAIN, "TRAIN is only ModeKey left"

        

        logging_dict = {k: v[1] for k, v in metrics.items()}
        logging_dict['loss'] = loss

        for k, v in metrics.items():
            tf.summary.scalar(k, v[1]) 

        if args.optimizer == 'adam':
            optimizer = tf.train.AdamOptimizer(args.learning_rate)
            train_op = optimizer.minimize(loss, global_step=tf.train.get_global_step())
            
        elif args.optimizer == 'adamw' or args.optimizer == 'lamb':
            do_update = tf.get_variable('do_update', shape=(), dtype=tf.float32, initializer=tf.constant_initializer(), aggregation=tf.VariableAggregation.ONLY_FIRST_REPLICA)
            if use_accumulation:
                training_hooks.append(GradientAccumulationHook(args.accumulation_steps, do_update))
            train_op = optimization.create_optimizer(
            loss, args.learning_rate, ITERATIONS, args.warmup_steps,
            False, args.optimizer, do_update=do_update, gradient_accumulation_multiplier=args.accumulation_steps)
        else:
            raise ValueError('Optimizer has to be "adam", "adamw" or "lamb"')
        
        if args.optimizer == 'adam':
            grads = optimizer.compute_gradients(loss)
            for grad, var in grads:
                if grad is not None:
                    tf.summary.histogram(var.op.name + '/gradients', grad)
        for var in tf.trainable_variables():
            tf.summary.histogram(var.name, var)
        logging_hook = tf.train.LoggingTensorHook(logging_dict, every_n_iter=args.accumulation_steps if args.accumulation_steps > 1 else 100)
        training_hooks.append(logging_hook)

        #train_op = count_labels(train_op, label_ids)
        

        return tf.estimator.EstimatorSpec(mode, loss=loss, train_op=train_op, training_hooks=training_hooks)
    return my_model_fn

def count_labels(train_op, label_ids):
    label_0 = tf_count(label_ids, 0)
    label_1 = tf_count(label_ids, 1)
    label_2 = tf_count(label_ids, 2)

    new_counts = tf.stack([label_0, label_1, label_2], axis=0)
    new_counts = tf.reshape(new_counts, [-1])

    example_counter = tf.get_variable('example_counter', shape=(3), dtype=tf.int32, initializer=tf.constant_initializer(), aggregation=tf.VariableAggregation.SUM)
    
    update_counter = tf.assign_add(example_counter, new_counts)

    update_counter = tf.Print(update_counter, [update_counter])

    train_op = tf.group(train_op, [update_counter])
    return train_op

ctrl_dependencies = []

def get_metrics(y_true, y_pred, target_names=target_names[args.dataset]):

    len_tn = len(target_names)
    nr_classes = tf.shape(y_true)[-1]
    y_true = tf.one_hot(y_true, depth=len_tn, dtype=tf.float32)
    y_pred = tf.one_hot(y_pred, depth=len_tn, dtype=tf.float32)
    
    assert_op = tf.debugging.assert_equal(nr_classes, len_tn, message=tf.strings.format("Number of classes, {}, must equal the number of target names, {}.", inputs=(nr_classes, len_tn)))


    ctrl_dependencies.append(assert_op)

    target_names = [tn.lower() for tn in target_names]
    
    metrics = {}
    f1s = [0, 0, 0]

    y_true = tf.cast(y_true, tf.float64)
    y_pred = tf.cast(y_pred, tf.float64)

    precisions = [0,0,0]
    recalls = [0,0,0]
    
    support = tf.count_nonzero(y_true, axis=0)
    # axis=None -> micro, axis=1 -> macro
    for i, axis in enumerate([None, 0]):
        TP = tf.count_nonzero(y_pred * y_true, axis=axis, dtype=tf.dtypes.float64)
        FP = tf.count_nonzero(y_pred * (y_true - 1), axis=axis, dtype=tf.dtypes.float64)
        FN = tf.count_nonzero((y_pred - 1) * y_true, axis=axis, dtype=tf.dtypes.float64)

        precision = tf.math.divide_no_nan(TP, (TP + FP))
        recall = tf.math.divide_no_nan(TP, (TP + FN))
        f1 = tf.math.divide_no_nan(2 * precision * recall, (precision + recall))

        precisions[i] = tf.metrics.mean(precision)
        recalls[i] = tf.metrics.mean(recall)
        f1s[i] = tf.metrics.mean(f1)

    weights = tf.reduce_sum(y_true, axis=0)
    weights /= tf.reduce_sum(weights)
    
    precisions[2] = tf.metrics.mean(tf.reduce_sum(precision * weights))
    recalls[2] = tf.metrics.mean(tf.reduce_sum(recall * weights))
    f1s[2] = tf.metrics.mean(tf.reduce_sum(f1 * weights))

    tot_supp = tf.reduce_sum(support)
    supports = tf.metrics.mean(tot_supp)
    supports = [supports for i in range(3)]

    precision = [tf.metrics.mean(precision[i]) for i in range(len_tn)]
    recall = [tf.metrics.mean(recall[i]) for i in range(len_tn)]
    f1 = [tf.metrics.mean(f1[i]) for i in range(len_tn)]
    support = [tf.metrics.mean(support[i]) for i in range(len_tn)]

    precision_dim = tf.shape(precision)[-1]

    for i, metric in enumerate(['precision', 'recall', 'f1', 'support']):
        for j, sublist in enumerate([target_names, ['micro-avg', 'macro-avg', 'weighted-avg']]):
            for k, label in enumerate(sublist):
                key = '{}_{}'.format(metric, label)
                lookup = [[precision, precisions], [recall, recalls], [f1, f1s], [support, supports]]
                metrics[key] = lookup[i][j][k]
    
    return metrics




# sampling parameters use it wisely 
oversampling_coef = 0.9 # if equal to 0 then oversample_classes() always returns 1
undersampling_coef = 0.9 # if equal to 0 then undersampling_filter() always returns True

def oversample_classes(example):
    """
    Returns the number of copies of given example
    """
    label_id = example['label_ids']
    # Fn returning negative class probability
    #def f1(i): return tf.constant(class_probabilities[args.dataset][i])
    #def f2(): return tf.cond(tf.math.equal(label_id, tf.constant(1)), lambda: f1(1), lambda: f1(2))

    #class_prob = tf.cond(tf.math.equal(label_id, tf.constant(0)), lambda: f1(0), f2)
    class_prob = tf.gather(class_probabilities[args.dataset], label_id)
    class_target_prob = tf.constant(1/num_labels[args.dataset])
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

    tf.logging.info("Oversampling label with repeat count: " + str(repeat_count + residual_acceptance))

    return repeat_count + residual_acceptance


def undersampling_filter(example):
    """
    Computes if given example is rejected or not.
    """
    label_id = example['label_ids']
    # Fn returning negative class probability
    #def f1(i): return tf.constant(class_probabilities[args.dataset][i])
    #def f2(): return tf.cond(tf.math.equal(label_id, tf.constant(1)), lambda: f1(1), lambda: f1(2))

    #class_prob = tf.cond(tf.math.equal(label_id, tf.constant(0)), lambda: f1(0), f2)
    class_prob = tf.gather(class_probabilities[args.dataset], label_id)
    class_target_prob = tf.constant(1/num_labels[args.dataset])

    prob_ratio = tf.cast(class_target_prob/class_prob, dtype=tf.float32)
    prob_ratio = prob_ratio ** undersampling_coef
    prob_ratio = tf.minimum(prob_ratio, 1.0)

    acceptance = tf.less_equal(tf.random_uniform([], dtype=tf.float32), prob_ratio)
    # predicate must return a scalar boolean tensor
    return acceptance

def log_prediction_on_test(classifier):
    sp = spm.SentencePieceProcessor(model_file=ALBERT_PRETRAINED_PATH + os.sep + '30k-clean.model')
    table = wandb.Table(columns=["Tweet", "Predicted Label", "True Label"])
    gold_labels = []
    predictions = []
    for pred in classifier.predict(input_fn=lambda: eval_input_fn(args.batch_size, test=True)):
        input_ids = pred['input_ids']
        gold = pred['gold']
        gold_labels.append(gold)
        prediction = pred['predictions']
        predictions.append(prediction)
        text = ''.join([sp.id_to_piece(id) for id in input_ids.tolist()]).replace('▁', ' ')
        text = re.sub('(<pad>)*$|(\[SEP\])|^(\[CLS\])', '', text)
        table.add_data(text, prediction, gold)

    print(classification_report(gold_labels, predictions, target_names=target_names[args.dataset], digits=8))
    print(confusion_matrix(gold_labels, predictions))
    wandb.log({"Predictions Test": table})

def log_prediction_on_dev(classifier):
    sp = spm.SentencePieceProcessor(model_file=ALBERT_PRETRAINED_PATH + os.sep + '30k-clean.model')
    table = wandb.Table(columns=["Tweet", "Predicted Label", "True Label"])
    gold_labels = []
    predictions = []
    for pred in classifier.predict(input_fn=lambda: eval_input_fn(args.batch_size)):
        input_ids = pred['input_ids']
        gold = pred['gold']
        gold_labels.append(gold)
        prediction = pred['predictions']
        predictions.append(prediction)
        text = ''.join([sp.id_to_piece(id) for id in input_ids.tolist()]).replace('▁', ' ')
        text = re.sub('(<pad>)*$|(\[SEP\])|^(\[CLS\])', '', text)
        table.add_data(text, prediction, gold)
    print(classification_report(gold_labels, predictions, target_names=target_names[args.dataset], digits=8))
    print(confusion_matrix(gold_labels, predictions))
    wandb.log({"Predictions Dev": table})

def decode_input_ids(input_ids, processor):
    text = ''.join([processor.id_to_piece(id) for id in input_ids.tolist()]).replace('▁', ' ')
    text = re.sub('(<pad>)*$|(\[SEP\])|^(\[CLS\])', '', text)
    return text

def get_sentence_piece_processor():
    return spm.SentencePieceProcessor(model_file=ALBERT_PRETRAINED_PATH + os.sep + '30k-clean.model')

def log_predictions(input_ids, predictions, gold, probabilities, name=""):
    sp = get_sentence_piece_processor()
    tweets = [decode_input_ids(input_id, processor=sp) for input_id in input_ids]
    if name == "Majority Voting":
        table = wandb.Table(columns=["Model Nr", "Tweet", "Predicted Label", "True Label", "Probabilities"])
        for model_nr, (pred_array, prob_array) in enumerate(zip(predictions, probabilities)):
            df = pd.DataFrame({'Tweets':tweets, 'Predictions':pred_array, 'Gold': gold, 'Probabilities': prob_array})
            df.to_pickle(os.path.join(model_dir, 'predictions_majority_voting_{}.pkl'.format(model_nr)))
            for tweet, pred, g, probs in zip(tweets, pred_array, gold, prob_array):
                table.add_data(model_nr, tweet, pred, g, probs)
    else:
        table = wandb.Table(columns=["Tweet", "Predicted Label", "True Label", "Probabilities"])
        df = pd.DataFrame({'Tweets': tweets, 'Predictions': predictions, 'Gold': gold, 'Probabilities': probabilities})
        df.to_pickle(os.path.join(model_dir, 'predictions_mean.pkl'))
        for tweet, pred, g, probs in zip(tweets, predictions, gold, probabilities):
            table.add_data(tweet, pred, g, probs)
    wandb.log({"Predictions {}".format(name): table})

def predict_tree():
    config_a = AlbertHateConfig.from_json_file('configs/top-a.json')
    classifier_a = tf.estimator.Estimator(
            model_fn=model_fn_builder(config=config_a),
            model_dir=os.path.abspath(config_a.model_dir)
            )
    pred_a = [pred for pred in classifier_a.predict(input_fn=lambda: eval_input_fn(128, test=args.test))]

    config_b = AlbertHateConfig.from_json_file('configs/top-b.json')
    classifier_b = tf.estimator.Estimator(
            model_fn=model_fn_builder(config=config_b),
            model_dir=os.path.abspath(config_b.model_dir)
            )
    pred_b = [pred for pred in classifier_b.predict(input_fn=lambda: eval_input_fn(128, test=args.test))]

    config_c = AlbertHateConfig.from_json_file('configs/top-c.json')
    classifier_c = tf.estimator.Estimator(
            model_fn=model_fn_builder(config=config_c),
            model_dir=os.path.abspath(config_c.model_dir)
            )
    
    pred_c = [pred for pred in classifier_c.predict(input_fn=lambda: eval_input_fn(32, test=args.test))]

    final_pred = []
    gold = []
    print("Predicting on converted")
    print("a(NOT) -> NONE | b(UNT) -> OFF | c(GRP) -> HATE | OFF")
    for i in range(len(pred_a)):
        gold.append(int(pred_a[i]['gold']))
        prediction_a = pred_a[i]['predictions']
        if int(prediction_a) == 0:
            final_pred.append(2)
            """
            prediction_b = pred_b[i]['predictions']
            if int(prediction_b) == 1:
                final_pred.append(2)
            else:
                prediction_c = pred_c[i]['predictions']
                if int(prediction_c) == 0:
                    final_pred.append(1)
                elif int(prediction_c) == 1:
                    final_pred.append(0)
                else:
                    final_pred.append(2)
            """
        else:
            prediction_b = pred_b[i]['predictions']
            if int(prediction_b) == 1:
                final_pred.append(1)
            else:
                prediction_c = pred_c[i]['predictions']
                if int(prediction_c) == 1:
                    final_pred.append(0)
                else:
                    final_pred.append(1)
    
    print(classification_report(gold, final_pred, target_names=target_names[args.dataset], digits=8))
    print(confusion_matrix(gold, final_pred))

def run_ensamble():
    with tf.gfile.GFile('best_ensamble.json', "r") as reader:
        text = reader.read()
    configs = [AlbertHateConfig.from_dict(cnfg) for cnfg in json.loads(text)]

    gold = []
    predictions = []
    probabilities = []
    input_ids = []
    for i, config in enumerate(configs):
        ws = tf.estimator.WarmStartSettings(ckpt_to_initialize_from=config.best_checkpoint, vars_to_warm_start='.*')
        classifier = tf.estimator.Estimator(
            model_fn=model_fn_builder(config=config),
            model_dir=model_dir,
            config=tf.estimator.RunConfig(train_distribute=tf.distribute.MirroredStrategy()),
            warm_start_from=ws
            )
        preds = [pred for pred in classifier.predict(input_fn=lambda: eval_input_fn(64, test=args.test))]
        if i == 0:
            gold = [int(p['gold']) for p in preds]
            input_ids = [p['input_ids'] for p in preds]
        # Predictions and probabilities are different between models, so these need to be added separately
        predictions.append([int(p['predictions']) for p in preds])
        probabilities.append([p['probabilities'] for p in preds])

        print("----------Results for model {}----------------".format(i))
        print(classification_report(gold, predictions[i], target_names=target_names[args.dataset], digits=8))
        print(confusion_matrix(gold, predictions[i]))
    
    final_pred = [np.argmax(np.bincount(preds)) for preds in np.transpose(np.asarray(predictions))]
    


    print("------- Results with majority voting -----------")
    print(classification_report(gold, final_pred, target_names=target_names[args.dataset], digits=8))
    print(confusion_matrix(gold, final_pred))

    log_predictions(input_ids, predictions, gold, probabilities, 'Majority Voting')

    probabilities = np.asarray(probabilities)

    final_pred = np.argmax(np.sum(probabilities, axis=0), axis=-1)
    print("------- Results with mean -----------")
    print(classification_report(gold, final_pred, target_names=target_names[args.dataset], digits=8))
    print(confusion_matrix(gold, final_pred))
    
    log_predictions(input_ids, final_pred, gold, np.sum(probabilities, axis=0)/len(configs), 'Mean')


if __name__ == "__main__":
    with tf.control_dependencies(ctrl_dependencies):
        if args.tree_predict:
            predict_tree()
        elif args.ensamble:
            run_ensamble()
        else:
            tf.debugging.set_log_device_placement(True)
            strategy = tf.distribute.MirroredStrategy()
            config = tf.estimator.RunConfig(train_distribute=strategy, save_checkpoints_steps=500)

            if args.wandb_run_path:
                tf.logging.info("Using WandB run path: {}".format(args.wandb_run_path))
                model_config = wandb.restore('model_config.json', run_path=args.wandb_run_path)
                model_config = AlbertHateConfig.from_json_file(model_config.name)
                ws = tf.estimator.WarmStartSettings(ckpt_to_initialize_from=model_config.model_dir, vars_to_warm_start='.*')
                if args.test:
                    model_dir = model_config.model_dir
                    tf.logging.info("Using model_dir from model_config: {}".format(model_dir))
            elif args.config_path:
                model_config = AlbertHateConfig.from_json_file(args.config_path)
                ws = tf.estimator.WarmStartSettings(ckpt_to_initialize_from=model_config.model_dir, vars_to_warm_start='.*')
                if args.test:
                    model_dir = model_config.model_dir
            elif args.init_checkpoint:
                ws = tf.estimator.WarmStartSettings(ckpt_to_initialize_from=args.init_checkpoint, vars_to_warm_start='.*')
                model_config = None
            else:
                ws = None
                model_config = None
                


            classifier = tf.estimator.Estimator(
                model_fn=model_fn_builder(model_config),
                model_dir=os.path.abspath(model_dir),
                config=config,
                warm_start_from=ws
                )

            if args.test:
                classifier.evaluate(input_fn=lambda:eval_input_fn(args.batch_size, test=True), steps=None)
                log_prediction_on_test(classifier)
            elif args.predict:
                log_prediction_on_dev(classifier)
            else:
                early_stopping = tf.estimator.experimental.stop_if_no_decrease_hook(classifier, metric_name="eval_loss", max_steps_without_decrease=1000, min_steps=1000)



                train_spec = tf.estimator.TrainSpec(input_fn=lambda: train_input_fn(args.batch_size), max_steps=ITERATIONS, hooks=[wandb.tensorflow.WandbHook(steps_per_log=500), early_stopping])
                eval_spec = tf.estimator.EvalSpec(input_fn=lambda:eval_input_fn(args.batch_size), steps=None)

                tf.estimator.train_and_evaluate(classifier, train_spec, eval_spec)

                #classifier.train(input_fn=lambda: train_input_fn(args.batch_size),
                #                    hooks=[wandb.tensorflow.WandbHook(steps_per_log=500)],
                #                    max_steps=ITERATIONS)

                log_prediction_on_test(classifier)

                classifier.evaluate(input_fn=lambda:eval_input_fn(args.batch_size, test=True), steps=None)

                if model_config is None:
                    model_config = AlbertHateConfig(linear_layers=args.linear_layers, 
                                                model_dir=model_dir,
                                                model_size=args.model_size,
                                                num_labels=args.num_labels,
                                                regression=args.regression,
                                                sequence_length=args.sequence_length,
                                                use_seq_out=args.use_seq_out)
                else:
                    model_config.model_dir = model_dir
                
                config_file = os.path.join(model_dir, 'model_config.json')
                
                with tf.gfile.GFile(config_file, "w") as writer:
                    writer.write(model_config.to_json_string())


            

    
