from mpi4py import MPI
import numpy as np
import pandas as pd
import os
from preprocessing import OlidProcessor
from tokenization import FullTokenizer

comm = MPI.COMM_WORLD
size = comm.Get_size()
rank = comm.Get_rank()

if rank == 0:
  processor = OlidProcessor()
  solid = processor.get_2020_dataframe(os.path.join('data', 'offenseval-2020'), 'a')

  solid = solid.iloc[:500]

  tweets = solid.tweet.to_numpy()
  labels = solid.subtask_a.numpy()

  tweets = np.array_split(tweets, size)
  labels = np.array_split(labels, size)

else:
  tweets = None
  labels = None

tokenizer = FullTokenizer(strip_handles=False, segment_hashtags=True, demojize=True, remove_url=True, remove_rt=True)
tokenize = np.vectorize(tokenizer.tokenize)
process_labels = np.vectorize(lambda label: 0 if label == 'NOT' else 1)

tweets = comm.scatter(tweets, root=0)
labels = comm.scatter(labels, root=0)

text_a = tokenize(tweets)
labels = process_labels(labels)

text_a = comm.gather(text_a, root=0)
labels = comm.gather(labels, root=0)

if rank == 0:
  ernie = pd.DataFrame({'text_a': text_a, 'labels': labels})
  print(ernie.head())
  print(ernie.shape)


