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
  #df = processor.get_2020_dataframe(os.path.join('data', 'offenseval-2020'), 'a')
  df = processor.get_2020_test_dataframe(os.path.join('data', 'offenseval-2020'), 'a')
  print("Read SOLID")

  tweets = df.tweet.to_numpy()
  labels = df.subtask_a.to_numpy()
  del df
  print("Splitting arrays")

  tweets = np.array_split(tweets, size)
  labels = np.array_split(labels, size)
  print("Arrays split")
else:
  tweets = None
  labels = None

tweets = comm.scatter(tweets, root=0)
labels = comm.scatter(labels, root=0)

print("{} got {} tweets".format(rank, len(tweets)))

print("{} getting tokenizer\n".format(rank))
tokenizer = FullTokenizer(strip_handles=False, segment_hashtags=True, demojize=True, remove_url=True, remove_rt=True)
tokenize = np.vectorize(tokenizer.tokenize)
process_labels = np.vectorize(lambda label: 0 if label == 'NOT' else 1)

print("{} tokenizing tweets\n".format(rank))
text_a = tokenize(tweets)
del tweets
print("{} processing labels\n".format(rank))
labels = process_labels(labels)

print("Rank {} of {} finished\n".format(rank, size))

text_a = comm.gather(text_a, root=0)
labels = comm.gather(labels, root=0)

if rank == 0:
  text_a = np.concatenate(text_a)
  labels = np.concatenate(labels)

  print(text_a.shape)
  print(labels.shape)
  solid_ernie = pd.DataFrame({'text_a': text_a, 'labels': labels})
  print(solid_ernie.shape)

  solid_ernie.to_csv('../ernie/data/solid/test.tsv', index=False, sep='\t')


