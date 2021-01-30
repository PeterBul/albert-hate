from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from run_albert_hate import run_cross_validation
import pandas as pd

if __name__ == "__main__":
  mic = []
  mac = []
  w = []
  for i in range(10):
    f1_micro, f1_macro, f1_weighted = run_cross_validation(i)
    mic.append(f1_micro)
    mac.append(f1_macro)
    w.append(f1_weighted)
  
  df = pd.DataFrame(
    {'f1_micro': mic,
     'f1_macro': mac,
     'f1_weighted': w 
    })
  
  df.to_csv('out/cross-val.csv')
    