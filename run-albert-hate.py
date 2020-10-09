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

# Control arguments
parser.add_argument('--init_checkpoint', type=str, default=None, help="Checkpoint to train from")
parser.add_argument('--model_dir', type=str, help='The name of the folder to save the model')
parser.add_argument('--predict', type=utils.str2bool, const=True, default=False, nargs='?')
parser.add_argument('--debug', type=utils.str2bool, const=True, default=False, nargs='?', help='Set flag to debug')
parser.add_argument('--test', type=utils.str2bool, const=True, default=False, nargs='?', help='Set flag to test')
parser.add_argument('--dataset', type=str, default='solid', help='The name of the dataset to be used. For now,  davidson or solid')
parser.add_argument('--task', type=str, default='a', help='Task a, b or c for solid/olid dataset')
parser.add_argument('--tree_predict', type=utils.str2bool, const=True, default=False, nargs='?')

# Arguments that aren't very important
parser.add_argument('--epochs', type=int, default=-1, help='Number of epochs to train. If not set, iterations are used instead.')
parser.add_argument('--dataset_length', type=int, default=-1, help='Length of dataset. Is needed to caluclate iterations if epochs is used.')

args = parser.parse_args()

use_accumulation = args.accumulation_steps > 1

print(args.__dict__)

wandb.init(
  project="albert-hate",
  config=args.__dict__,
  sync_tensorboard=True
)

model_dir = wandb.run.dir if args.model_dir is None else args.model_dir

config = wandb.config


class AlbertHateConfig(object):
    def __init__(self,
                    num_labels=num_labels[args.dataset],
                    batch_size=32,
                    linear_layers=0,
                    model_size='base',
                    use_seq_out=True,
                    regression=False,
                    sequence_length=128,
                    model_dir=None,
                    task=None
                    ):
        self.num_labels = num_labels
        self.batch_size = batch_size
        self.linear_layers = linear_layers
        self.model_size = model_size
        self.use_seq_out = utils.str2bool(use_seq_out)
        self.regression = utils.str2bool(regression)
        self.sequence_length = sequence_length
        self.model_dir = model_dir
        self.task = task
    
    @classmethod
    def from_dict(cls, json_object):
        """Constructs a `AlbertHateConfig` from a Python dictionary of parameters."""
        config = AlbertHateConfig()
        for (key, value) in six.iteritems(json_object):
            if isinstance(config.__dict__[key], bool):
                value = utils.str2bool(value)
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
    FILE_TRAIN = PATH_DATASET + os.sep + args.dataset + os.sep + 'train-' + str(args.sequence_length) + '.tfrecords'
    FILE_DEV = PATH_DATASET + os.sep + args.dataset + os.sep + 'dev-' + str(args.sequence_length) + '.tfrecords'
    FILE_TEST = PATH_DATASET + os.sep + args.dataset + os.sep + 'test' + os.sep + 'test-' + str(args.sequence_length) + '.tfrecords'
elif args.dataset == 'solid':
    FILE_TRAIN = PATH_DATASET + os.sep + 'solid' + os.sep + 'task-' + args.task + os.sep + 'solid-' + str(args.sequence_length) + '.tfrecords'
    FILE_DEV = PATH_DATASET + os.sep + 'olid' + os.sep + 'task-' + args.task + os.sep + 'olid-' + str(args.sequence_length) + '.tfrecords'
    FILE_TEST = PATH_DATASET + os.sep + 'solid' + os.sep + 'task-' + args.task + os.sep + 'test' + os.sep + 'solid-' + str(args.sequence_length) + '.tfrecords'
elif args.dataset =='converted':
    FILE_TRAIN = PATH_DATASET + os.sep + 'converted' + os.sep + 'solid-' + str(args.sequence_length) + '.tfrecords'
    FILE_DEV = PATH_DATASET + os.sep + 'converted' + os.sep + 'olid-' + str(args.sequence_length) + '.tfrecords'
    FILE_TEST = PATH_DATASET + os.sep + 'converted' + os.sep + 'test-' + str(args.sequence_length) + '.tfrecords'
else:
    #FILE_TRAIN = PATH_DATASET + os.sep + 'olid-2020-full-'+ ("reg-" if args.regression else "") + str(args.sequence_length) + '.tfrecords'
    #FILE_DEV = PATH_DATASET + os.sep + 'olid-2019-full-' + str(args.sequence_length) + '.tfrecords'
    #FILE_TEST = PATH_DATASET + os.sep + 'test-2020-' + str(args.sequence_length) + '.tfrecords'
    raise ValueError("Dataset not supported")

if args.dataset in ('solid', 'olid'):
    args.__dict__['dataset'] = args.dataset + '_' + args.task

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

def train_input_fn(batch_size):
    ds = tf.data.TFRecordDataset(FILE_TRAIN)
    
    ds = ds.map(utils.read_tfrecord_builder(is_training=True, seq_length=args.sequence_length, regression=args.regression))
    if args.use_resampling:
        tf.logging.info("Resampling dataset with oversampling and undersampling")
        ds = ds.flat_map(
            lambda x: tf.data.Dataset.from_tensors(x).repeat(oversample_classes(x))
        )
        #ds = ds.filter(undersampling_filter)
    ds = ds.repeat().shuffle(2048).batch(batch_size, drop_remainder=False)
    return ds

def eval_input_fn(batch_size, test=False):
    if not test:
        tf.logging.info('Using DEV dataset: {}'.format(FILE_DEV))
        ds = tf.data.TFRecordDataset(FILE_DEV)
    else:
        tf.logging.info('Using TEST dataset: {}'.format(FILE_TEST))
        ds = tf.data.TFRecordDataset(FILE_TEST)
    assert batch_size is not None, "batch_size must not be None"
    ds = ds.map(utils.read_tfrecord_builder(is_training=False, seq_length=args.sequence_length)).batch(batch_size)
    return ds


def model_fn_builder(config=None):    
    def my_model_fn(features, labels, mode):
        

        training_hooks = []

        albert_pretrained_path = 'albert_' + config.model_size

        albert_config = modeling.AlbertConfig.from_json_file(albert_pretrained_path + os.sep + 'albert_config.json')
        if args.albert_dropout is not None:
            albert_config.hidden_dropout_prob = args.albert_dropout
        #albert_config.attention_probs_dropout_prob = args.dropout

        
        input_ids = features['input_ids']
        input_mask = features['input_mask']
        segment_ids = features['segment_ids']
        

        is_training = mode == tf.estimator.ModeKeys.TRAIN

        if config.regression and is_training:
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
                                                                            num_labels=config.num_labels,
                                                                            use_one_hot_embeddings=False,
                                                                            task_name='hsc', 
                                                                            hub_module=None, 
                                                                            linear_layers=config.linear_layers,
                                                                            regression=config.regression,
                                                                            use_sequence_output=config.use_seq_out)
        
        tvars = tf.trainable_variables()
        initialized_variable_names = {}

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

        return tf.estimator.EstimatorSpec(mode, loss=loss, train_op=train_op, training_hooks=training_hooks)
    return my_model_fn

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
    for pred in classifier.predict(input_fn=lambda: eval_input_fn(args.batch_size, test=True)):
        input_ids = pred['input_ids']
        gold = pred['gold']
        prediction = pred['predictions']
        text = ''.join([sp.id_to_piece(id) for id in input_ids.tolist()]).replace('▁', ' ')
        text = re.sub('(<pad>)*$|(\[SEP\])|^(\[CLS\])', '', text)
        table.add_data(text, prediction, gold)
    wandb.log({"Predictions Test": table})

def log_prediction_on_dev(classifier):
    sp = spm.SentencePieceProcessor(model_file=ALBERT_PRETRAINED_PATH + os.sep + '30k-clean.model')
    table = wandb.Table(columns=["Tweet", "Predicted Label", "True Label"])
    for pred in classifier.predict(input_fn=lambda: eval_input_fn(args.batch_size)):
        input_ids = pred['input_ids']
        gold = pred['gold']
        prediction = pred['predictions']
        text = ''.join([sp.id_to_piece(id) for id in input_ids.tolist()]).replace('▁', ' ')
        text = re.sub('(<pad>)*$|(\[SEP\])|^(\[CLS\])', '', text)
        table.add_data(text, prediction, gold)
    wandb.log({"Predictions Dev": table})


def predict_tree():
    test = True
    sp = spm.SentencePieceProcessor(model_file=ALBERT_PRETRAINED_PATH + os.sep + '30k-clean.model')
    config_a = AlbertHateConfig.from_json_file('configs/top-a.json')
    classifier_a = tf.estimator.Estimator(
            model_fn=model_fn_builder(config=config_a),
            model_dir=os.path.abspath(config_a.model_dir)
            )
    pred_a = [pred for pred in classifier_a.predict(input_fn=lambda: eval_input_fn(config_a.batch_size))]

    config_b = AlbertHateConfig.from_json_file('configs/top-b.json')
    classifier_b = tf.estimator.Estimator(
            model_fn=model_fn_builder(config=config_b),
            model_dir=os.path.abspath(config_b.model_dir)
            )
    pred_b = [pred for pred in classifier_b.predict(input_fn=lambda: eval_input_fn(config_b.batch_size))]

    config_c = AlbertHateConfig.from_json_file('configs/top-c.json')
    classifier_c = tf.estimator.Estimator(
            model_fn=model_fn_builder(config=config_c),
            model_dir=os.path.abspath(config_c.model_dir)
            )
    
    pred_c = [pred for pred in classifier_c.predict(input_fn=lambda: eval_input_fn(config_c.batch_size))]

    final_pred = []
    gold = []
    print("Predicting on davidson")
    print("a(NOT) -> NONE | b(UNT) -> OFF | c(OTH) -> OFF | HATE")
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
    
    print(classification_report(gold, final_pred, target_names=target_names[args.dataset]))
    print(confusion_matrix(gold, final_pred))


if __name__ == "__main__":
    with tf.control_dependencies(ctrl_dependencies):
        if args.tree_predict:
            predict_tree()
        else:
            tf.debugging.set_log_device_placement(True)
            strategy = tf.distribute.MirroredStrategy()
            config = None if args.test else tf.estimator.RunConfig(train_distribute=strategy, save_checkpoints_steps=500)

            if args.init_checkpoint:
                ws = tf.estimator.WarmStartSettings(ckpt_to_initialize_from=args.init_checkpoint, vars_to_warm_start='.*')
            else:
                ws = None
            

            classifier = tf.estimator.Estimator(
                model_fn=model_fn_builder(),
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
                #early_stopping = tf.estimator.experimental.stop_if_no_decrease_hook(classifier, metric_name="eval_loss", max_steps_without_decrease=10000, min_steps=1000)
                train_spec = tf.estimator.TrainSpec(input_fn=lambda: train_input_fn(args.batch_size), max_steps=ITERATIONS, hooks=[wandb.tensorflow.WandbHook(steps_per_log=500)])
                eval_spec = tf.estimator.EvalSpec(input_fn=lambda:eval_input_fn(args.batch_size), steps=None)

                tf.estimator.train_and_evaluate(classifier, train_spec, eval_spec)

                log_prediction_on_dev(classifier)

                classifier.evaluate(input_fn=lambda:eval_input_fn(args.batch_size, test=True), steps=None)


            

    
