from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import wandb
import os
from run_albert_hate import AlbertHateConfig, model_fn_builder, eval_input_fn
import tensorflow as tf
from sklearn.metrics import classification_report
from constants import target_names
from print_metrics import flatten_classification_report
import pandas as pd
import argparse

api = wandb.Api()

seq_length = 128
PATH_DATASET= 'data' + os.sep + 'tfrecords'


def get_file_from_ds(dataset):
  if dataset in ('davidson', 'converted', 'founta'):
    return PATH_DATASET + os.sep + dataset + os.sep + 'test-' + str(seq_length) + '.tfrecords'
  elif dataset in ('solid_a', 'solid_b', 'solid_c'):
    return PATH_DATASET + os.sep + 'solid' + os.sep + 'task-' + dataset.split('_')[-1] + os.sep + 'test' + os.sep + 'solid-' + str(seq_length) + '.tfrecords'
  elif dataset == 'founta-converted':
    return os.path.join(PATH_DATASET, 'founta', 'conv', 'test-' + str(seq_length) + '.tfrecords')
  elif dataset == 'founta/isaksen':
    return os.path.join(PATH_DATASET, 'founta', 'isaksen', 'test-' + str(seq_length) + '.tfrecords')
  elif dataset == 'founta/isaksen/spam':
    return os.path.join(PATH_DATASET, 'founta', 'isaksen', 'spam', 'test-' + str(seq_length) + '.tfrecords')
  elif dataset == 'combined':
    return os.path.join(PATH_DATASET, 'combined', 'test-' + str(seq_length) + '.tfrecords')
  else:
      raise ValueError("Dataset not supported")

def evaluate(run_id, dataset):
  test_set = get_file_from_ds(dataset)
  tf.logging.info("Using dataset file: {}".format(test_set))
  tf.logging.info("Using WandB run id: {}".format(run_id))

  run = api.run('petercbu/albert-hate/{}'.format(run_id))
  
  root = 'tmp'

  if not os.path.exists(root):
    os.makedirs(root)

  model_config = run.file('model_config.json')

  model_config.download(root=root, replace=True)

  model_config = AlbertHateConfig.from_json_file(os.path.join('tmp', 'model_config.json'))
  model_dir = model_config.model_dir
  if "files" in os.listdir(model_dir):
    model_dir = os.path.join(model_dir, 'files')
  ws = tf.estimator.WarmStartSettings(ckpt_to_initialize_from=model_dir, vars_to_warm_start='.*')
  
  tf.logging.info("Using model_dir from model_config: {}".format(model_dir))

  config = tf.estimator.RunConfig(train_distribute=tf.distribute.MirroredStrategy())

  classifier = tf.estimator.Estimator(
                model_fn=model_fn_builder(model_config),
                model_dir=os.path.abspath(model_dir),
                config=config,
                warm_start_from=ws
                )
  
  gold_labels = []
  predictions = []
  for pred in classifier.predict(input_fn=lambda: eval_input_fn(32, test=True, test_set=test_set, seq_length=128)):
    #mapping = {0:1,1:0,2:2,3:2}
    #if convert:
    #  prediction = mapping[prediction]

    gold = pred['gold']
    gold_labels.append(gold)
    prediction = pred['predictions']
    predictions.append(prediction)

  return classification_report(gold_labels, predictions, target_names=target_names[dataset], digits=8)

def batch_evaluate(run_ids, datasets):
  rows = []
  columns = None
  for dataset in datasets:
    for run_id in run_ids:
      cr = evaluate(run_id, dataset)
      print(run_id)
      print(cr)
      columns, row = flatten_classification_report(cr, name='{} {}'.format(run_id, dataset), return_columns=True)
      rows.append(row)
  
  if columns:
    df = pd.DataFrame(rows, columns=columns)
    df.to_csv("./tmp/eval.csv")


if __name__ == "__main__":
  #parser = argparse.ArgumentParser()

  #parser.add_argument('--run_ids', type=str, nargs='+', required=True)
  #parser.add_argument('--test_dataset', type=str, required=True)

  #args = parser.parse_args()

  run_ids = ['15wivgk8']
  datasets = 'davidson founta/isaksen'.split()

  batch_evaluate(run_ids, datasets)
  
