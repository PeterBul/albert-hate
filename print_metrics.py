import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, classification_report

def print_metrics(true, pred, target_names):
  print(classification_report(true, pred, target_names=target_names, digits=8))
  print(confusion_matrix(true, pred))

def get_classification_report(true, pred, target_names):
  return classification_report(true, pred, target_names=target_names, digits=8)



def flatten_classification_report(classification_report, name, return_columns=False):
  report = classification_report.split('\n')

  num_labels = len(report) - 7
  target_names = [report[i].split()[0] for i in range(2,2+num_labels)]


  columns = ["{}_{}".format(label, metric) for label in target_names + ['weighted-avg', 'macro-avg'] for metric in ['precision', 'recall', 'f1']]
  columns.append('accuracy')
  columns = ['name'] + columns
  row = [name]
  for i in range(2, 2 + num_labels):
    row_tmp = report[i].split()
    row += row_tmp[1:4]
  
  for i in range(-2, -4, -1):
    row_tmp = report[i].split()
    row += row_tmp[2:5]
  
  row.append(report[-4].split()[1])

  if return_columns:
    return columns, row
  else:
    return row
