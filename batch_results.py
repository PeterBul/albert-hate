from os import replace
import wandb
import pickle
import pandas as pd
from pathlib import Path
from print_metrics import flatten_classification_report


def get_classification_reports(path):
  result = []

  with open(path, 'r') as f:
    for line in f:
      if line.startswith("wandb: Run data is"):
        line = line.strip()
        model_dir = line.split()[-1]
        wandb_id = model_dir.split('-')[-1]
        wandb.restore('classification_report.pkl', replace=True, run_path='albert-hate/runs/' + wandb_id, root='./tmp')
        with open('./tmp/classification_report.pkl', 'rb') as handle:
          columns, cr = flatten_classification_report(pickle.load(handle), name=path, return_columns=True)
          result.append(cr)
          print(cr)
  return columns, result
        

if __name__ == "__main__":
  
  pathlist = Path('../slurm-jobs/slurm-out/').glob('*.err')
  rows = []
  for i, path in enumerate(pathlist):
    path_str = str(path)
    if path_str.endswith('2508666.err'):
      continue
    print(path_str)
    columns, result = get_classification_reports(path_str)
    rows = rows + result
  df = pd.DataFrame(rows, columns=columns)
  df.to_csv("./tmp/eval.csv")
    
  
  
