from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


import os
import pandas as pd
import numpy as np
import tensorflow as tf
from tqdm import tqdm
from collections import Counter, OrderedDict
from tokenization import TweetSpTokenizer
from sklearn.model_selection import train_test_split


class OlidExample(object):
  """A single training/test example for the OLID dataset.
  
  """

  def __init__(self,
               guid,
               tweet,
               label):
    self.guid = guid
    self.tweet = tweet
    self.label = label
    self._set_label_id(label)
    

  def __str__(self):
    return self.__repr__()

  def __repr__(self):
    s = ""
    s += "guid: {}".format(self.guid)
    s += ", tweet: {}".format(self.tweet)
    s += ", label: {}".format(self.label)
    return s
  
  def _set_label_id(self, label):
    if label == "NOT":
      self.label_id = 0
    elif label == "OFF":
      self.label_id = 1
    elif label == "TIN":
      self.label_id = 0
    elif label == "UNT":
      self.label_id = 1
    elif label == "IND":
      self.label_id = 0
    elif label == "GRP":
      self.label_id = 1
    elif label == "OTH":
      self.label_id =2
    else:
      raise ValueError("Label has to be NOT, OFF, TIN, UNT, IND; GRP or OTH")


class OlidRegExample(object):
  def __init__(self,
               guid,
               tweet,
               average):
    self.guid = guid
    self.tweet = tweet
    self.average = average

  def __str__(self):
    return self.__repr__()

  def __repr__(self):
    s = ""
    s += "guid: {}".format(self.guid)
    s += ", tweet: {}".format(self.tweet)
    s += ", average: {}".format(self.average)
    return s
  

class OlidProcessor(object):
  """Processor for OLID Dataset"""
  def __init__(self, do_lower_case=True, dev_fraction=0.2, random_state=42):
    self.do_lower_case = do_lower_case
    self.dev_fraction = dev_fraction
    self.random_state = random_state
  
  def get_train_examples(self, data_dir):
    """Gets a collection of `OlidExample`s for the training set"""
    df = pd.read_csv(os.path.join(data_dir, 'olid-training-v1.0.tsv'), sep='\t')
    if self.dev_fraction > 0:
      train, _ = train_test_split(df, test_size=self.dev_fraction, random_state=self.random_state)
    else:
      train = df
    return self._create_examples(train, 'a')
    
  
  def get_dev_examples(self, data_dir):
    df = pd.read_csv(os.path.join(data_dir, 'olid-training-v1.0.tsv'), sep='\t')
    if self.dev_fraction <= 0:
      raise ValueError(
          "This processor does not support development set. Set dev_fraction "
          "> 0."
      )
    _, dev = train_test_split(df, test_size=self.dev_fraction, random_state=self.random_state)
    return self._create_examples(dev, 'a')

  def get_test_examples(self, data_dir, task='a'):
    df_test = pd.read_csv(
        os.path.join(data_dir, 'testset-levela.tsv'), sep='\t')
    df_labels = pd.read_csv(
        os.path.join(data_dir, 'labels-levela.csv'),
        header=None, names=['id', 'subtask_a'])
    return self._create_examples(df_test.join(df_labels.set_index('id'), on='id'), task)

  def get_2020_test_examples(self, data_dir, task='a'):
    df_test = pd.read_csv(
        os.path.join(data_dir, 'test_' + task + '_tweets.tsv'), sep='\t'
    )
    df_labels = pd.read_csv(
        os.path.join(data_dir, 'gold' + os.path.sep + 'test_' + task + '_labels.csv'),
        header=None, names=['id', 'subtask_' + task]
    )
    return self._create_examples(df_test.join(df_labels.set_index('id'), on='id'), task)
  
  def get_2019_examples(self, data_dir, task):
    df = pd.read_csv(os.path.join(data_dir, 'olid-training-v1.0.tsv'), sep='\t')
    if task == 'b':
      df.dropna(subset=["subtask_b"], inplace=True)
    elif task == 'c':
      df.dropna(subset=["subtask_c"], inplace=True)
    return self._create_examples(df, task)

  def get_2020_examples(self, data_dir, task):
    assert task == 'a' or task == 'b' or task == 'c', "Task has to be a, b or c"
    df = pd.read_csv(os.path.join(data_dir, 'task-' + task + os.sep + 'task_' + task + '_distant.tsv'), sep='\t')
    print("Dataset read")
    if task == 'a':
      df['subtask_a'] = df.average.map(lambda avg: 'OFF' if avg > 0.5 else 'NOT')
    elif task == 'b':
      df['subtask_b'] = df.average.map(lambda avg: 'UNT' if avg > 0.5 else 'TIN')
    elif task == 'c':
      df['subtask_c'] = df.apply(lambda row: self._map_c_label(row.average_ind, row.average_grp, row.average_oth), axis=1)
    else:
      raise ValueError("Task has to be a, b or c")
    print("Dataset mapped")
    df.rename(columns = {'text':'tweet'}, inplace = True)
    print("Dataset column renamed")
    return self._create_examples(df, task)
  
  def get_2020_reg_examples(self, data_dir):
    df_a_20 = pd.read_csv(os.path.join(data_dir, 'task-a' + os.sep + 'task_a_distant.tsv'), sep='\t')
    print("Dataset read")
    df_a_20.rename(columns = {'text':'tweet'}, inplace = True)
    print("Dataset column renamed")
    return self._create_reg_examples(df_a_20)

  def create_tf_records(self, tokenizer, task):
    file_based_convert_examples_to_features(self.get_2020_examples('/content/drive/My Drive/prosjektoppgave/data/offenseval-2020', task), 128, tokenizer, '/content/drive/My Drive/masters_thesis/data/tfrecords/task-{}/new-solid-128.tfrecords'.format(task), task)
    file_based_convert_examples_to_features(self.get_2019_examples('/content/drive/My Drive/prosjektoppgave/data/offenseval-2019/', task), 128, tokenizer, '/content/drive/My Drive/masters_thesis/data/tfrecords/task-{}/new-olid-128.tfrecords'.format(task), task)
    file_based_convert_examples_to_features(self.get_2020_test_examples('/content/drive/My Drive/prosjektoppgave/data/offenseval-2020/task-{}/'.format(task)), 128, tokenizer, '/content/drive/My Drive/masters_thesis/data/tfrecords/task-{}/test/new-solid-128.tfrecords'.format(task), task)



  def get_labels(self):
    return ["NOT", "OFF"]

  def _create_examples(self, df, task):
    examples = []
    tf.logging.info("Dropping duplicates")
    df = df.drop_duplicates(subset='tweet')
    tf.logging.info("Dropping duplicates finished")
    pbar = tqdm(total=len(df.index))
    for index, row in df.iterrows():
      if task == 'a':
        label = row.subtask_a
      elif task == 'b':
        label = row.subtask_b
      elif task == 'c':
        label = row.subtask_c
      else:
        raise ValueError("Task has to be `a`, `b` or `c`")
      examples.append(
          OlidExample(
              row.id,
              row.tweet,
              label

          )
      )
      pbar.update(1)
    return examples
  
  def _create_reg_examples(self, df):
    examples = []
    pbar = tqdm(total=len(df.index))
    for index, row in df.iterrows():
      examples.append(
          OlidRegExample(
              row.id,
              row.tweet,
              row.average
          )
      )
      pbar.update(1)
    return examples

  def _map_c_label(self, ind, grp, oth):
    label = 'IND'
    high = ind
    if grp > high:
      label = 'GRP'
      high = grp
    if oth > high:
      label = 'OTH'
      high = oth
    return label

def convert_single_example(example_index, example, stats, max_seq_length, tokenizer, reg, task):
  tokenized_tweet = tokenizer.tokenize(example.tweet)
  if len(tokenized_tweet) > max_seq_length - 2:
    tokenized_tweet = tokenized_tweet[0:(max_seq_length - 2)]
  tokens = []
  segment_ids = []
  tokens.append("[CLS]")
  segment_ids.append(0)
  for token in tokenized_tweet:
    tokens.append(token)
    segment_ids.append(0)
  tokens.append("[SEP]")
  segment_ids.append(0)

  input_ids = tokenizer.convert_tokens_to_ids(tokens)

  stats['avg_len'] += len(input_ids)/stats['len_examples']
  stats['seq_len_dist'][len(input_ids)] += 1
  if len(input_ids) > stats['max_len']:
    stats['max_len'] = len(input_ids)

  input_mask = [1] * len(input_ids)

  while len(input_ids) < max_seq_length:
    input_ids.append(0)
    input_mask.append(0)
    segment_ids.append(0)
  
  assert len(input_ids) == max_seq_length
  assert len(input_mask) == max_seq_length
  assert len(segment_ids) == max_seq_length

  if not reg:
    if task == 'a':
      if example.label == "NOT":
        label_id = 0
        stats['not_count'] += 1
      elif example.label == "OFF":
        label_id = 1
        stats['off_count'] += 1
      else:
        raise ValueError(
            "Label is not `NOT` or `OFF` on example with "
            "index: {} and guid: {}".format(example_index, example.guid)
        )
    elif task == 'b':
      if example.label == 'TIN':
        label_id = 0
      elif example.label == 'UNT':
        label_id = 1
      else:
        raise ValueError(
            "Label is not `TIN` or `UNT` on example with "
            "index: {} and guid: {}. Make sure to remove NULL values".format(example_index, example.guid)
        )
    elif task == 'c':
      if example.label == 'IND':
        label_id = 0
      elif example.label == 'GRP':
        label_id = 1
      elif example.label == 'OTH':
        label_id = 2
      else:
        raise ValueError(
            "Label is not `IND`, `GRP` or `OTH` on example with "
            "index: {} and guid: {}. Make sure to remove NULL values".format(example_index, example.guid)
        )
    else:
      raise ValueError("Task {} not valid or implemented yet.".format(task))

  if example_index < 20:
    tf.logging.info("*** Example ***")
    tf.logging.info("guid: {}".format(example.guid))
    tf.logging.info("tokens: {}".format(" ".join(tokens)))
    tf.logging.info("input_ids: {}".format(" ".join([str(x) for x in input_ids])))
    tf.logging.info("input_mask: {}".format(" ".join([str(x) for x in input_mask])))
    tf.logging.info("segment_ids: {}".format(" ".join([str(x) for x in segment_ids])))
    if not reg:
      tf.logging.info("label: %s (id = %d)" % (example.label, label_id))
    else:
      tf.logging.info("average: {}".format(example.average))
  
  if not reg:
    feature = InputFeatures(
        input_ids=input_ids,
        input_mask=input_mask,
        segment_ids=segment_ids,
        label_id=label_id,
        is_real_example=True
    )
  else:
    feature = InputRegFeatures(
        input_ids=input_ids,
        input_mask=input_mask,
        segment_ids=segment_ids,
        average=example.average,
        is_real_example=True
    )
  return feature

class InputFeatures(object):
  """A single set of features of data."""

  def __init__(self,
               input_ids,
               input_mask,
               segment_ids,
               label_id,
               guid=None,
               example_id=None,
               is_real_example=True):
    self.input_ids = input_ids
    self.input_mask = input_mask
    self.segment_ids = segment_ids
    self.label_id = label_id
    self.example_id = example_id
    self.guid = guid
    self.is_real_example = is_real_example

class InputRegFeatures(object):
  def __init__(self,
               input_ids,
               input_mask,
               segment_ids,
               average,
               guid=None,
               example_id=None,
               is_real_example=True):
    self.input_ids = input_ids
    self.input_mask = input_mask
    self.segment_ids = segment_ids
    self.average = average
    self.example_id = example_id
    self.guid = guid
    self.is_real_example = is_real_example

def file_based_convert_examples_to_features(
    examples, max_seq_length, tokenizer, output_file, task, reg=False):
  
  writer = tf.python_io.TFRecordWriter(output_file)

  stats = {'avg_len': 0, 'len_examples': len(examples), 'max_len': 0, 'not_count': 0, 'off_count': 0, 'seq_len_dist': Counter()}

  for (ex_index, example) in enumerate(examples):
    if ex_index % 10000 == 0:
      tf.logging.info("Writing example %d of %d" % (ex_index, len(examples)))

    feature = convert_single_example(ex_index, example, stats, max_seq_length, 
                                     tokenizer, reg=reg, task=task)
    
    def _int64_feature(values):
      f = tf.train.Feature(int64_list=tf.train.Int64List(value=list(values)))
      return f
    
    def _float_feature(values):
      f = tf.train.Feature(float_list=tf.train.FloatList(value=list(values)))
      return f
    
    features = OrderedDict()
    features["input_ids"] = _int64_feature(feature.input_ids)
    features["input_mask"] = _int64_feature(feature.input_mask)
    features["segment_ids"] = _int64_feature(feature.segment_ids)
    if not reg:
      features["label_ids"] = _int64_feature([feature.label_id])
    else:
      features["average"] = _float_feature([feature.average])
    features["is_real_example"] = _int64_feature(
        [int(feature.is_real_example)])
    tf_example = tf.train.Example(features=tf.train.Features(feature=features))
    writer.write(tf_example.SerializeToString())
  writer.close()
  tf.logging.info('Average sequence length: {}'.format(stats['avg_len']))
  tf.logging.info('Number of examples: {}'.format(stats['len_examples']))
  tf.logging.info('Max length: {}'.format(stats['max_len']))
  tf.logging.info('Label distribution: NOT: {}, OFF: {}'.format(stats['not_count'], stats['off_count']))
  return stats


class Example(object):
  """A single training/test example.
  
  """

  def __init__(self,
               guid,
               tweet,
               label_id):
    self.guid = guid
    self.tweet = tweet
    if isinstance(label_id, int):
      self.label_id = label_id
      self.label = self._set_label(label_id)
    elif isinstance(label_id, str):
      self.label = label_id
      self.label_id = self._set_label_id(label_id)
    else:
      raise ValueError("label_id has to be string or int")

  def __str__(self):
    return self.__repr__()

  def __repr__(self):
    s = ""
    s += "guid: {}".format(self.guid)
    s += ", tweet: {}".format(self.tweet)
    s += ", label: {}".format(self.label)
    return s

  def _set_label(self, label_id):
    """Function that maps label_id to the label that is associated with that id.
    This is different from dataset to dataset, so has to be implemented.

    Parameters:
    label_id (int): Label id of example
    
    Returns:
    str: String label describing the label_id
    """
    raise NotImplementedError
  
  def _set_label_id(self, label):
    """ Function that maps label to label_id associated with label.

    Parameters:
    label (str): Label to be mapped to in

    Returns:
    int: Numeric label id representing label
    """
    raise NotImplementedError


class Processor(object):
  """Processor"""
  def __init__(self, dataset_name, do_lower_case=True, dev_fraction=0.2, test_fraction=0.2, random_state=42, keep_df=True):
    assert dev_fraction + test_fraction < 1, "Total of dev and test fraction has to be less than 1" 
    self.dataset_name = dataset_name
    self.do_lower_case = do_lower_case
    self.dev_fraction = dev_fraction
    self.test_fraction = test_fraction
    self.train_fraction = 1 - dev_fraction - test_fraction
    self.random_state = random_state
    self.keep_df = keep_df
    self.labels = self.get_labels()

    self._create_output_folder()
    if keep_df:
      self.df = self._read_df()
  
  
  def get_train_dev_test_examples(self):
    """Gets a collection for the training, dev and test set"""
    df = self.df if self.keep_df else self._read_df()
    if self.dev_fraction > 0 and self.test_fraction > 0:
      train, dev, test = np.split(df.sample(frac=1, random_state=self.random_state), 
                                  [int(self.train_fraction*len(df)), 
                                   int((self.train_fraction + self.dev_fraction)*len(df))])
      print("Train: {}, dev: {}, test: {}".format(len(train), len(dev), len(test)))
    else:
      raise ValueError("The processor doesn't support not using dev and test set")
    return self._create_examples(train), self._create_examples(dev), self._create_examples(test)

  def create_tfrecords(self, sequence_length, tokenizer=None):
    if tokenizer is None:
      tokenizer = TweetSpTokenizer()
    train, dev, test = self.get_train_dev_test_examples()
    self._create_tfrecord(train, 'train', sequence_length, tokenizer)
    self._create_tfrecord(dev, 'dev', sequence_length, tokenizer)
    self._create_tfrecord(test, 'test', sequence_length, tokenizer)

  def _create_output_folder(self):
    self.output_folder = '/content/drive/My Drive/masters_thesis/data/tfrecords/{}'.format(self.dataset_name)
    if not os.path.exists(self.output_folder):
      os.makedirs(self.output_folder)

  def _create_tfrecord(self, examples, name, sequence_length, tokenizer):
    tf.logging.info("Creating tfrecord: {}".format(name))
    output_file = os.path.join(self.output_folder, '{}-{}.tfrecords'.format(name, sequence_length))
    self._file_based_convert_examples_to_features(examples, sequence_length, tokenizer, output_file)

  def _create_examples(self, df):
    examples = []
    pbar = tqdm(total=len(df.index))
    for index, row in df.iterrows():
      examples.append(self._get_example_from_row(row))
      pbar.update(1)
    return examples
  
  def _get_example_from_row(self, row):
    """Create an example from a row of the pandas dataset.
    
    Parameters:
    row (pandas.core.series.Series): A row of a pandas dataframe

    Returns:
    Example: An example suited for the dataset the processor is meant for.

    """
    raise NotImplementedError
  

  def _read_df(self):
    """
    returns dataframe of dataset
    """
    raise NotImplementedError
  

  def get_labels(self):
    """
    Returns:
    list: List of labels
    """
    raise NotImplementedError



  def _convert_single_example(self, example_index, example, stats, max_seq_length, tokenizer):
    tokenized_tweet = tokenizer.tokenize(example.tweet)
    if len(tokenized_tweet) > max_seq_length - 2:
      tokenized_tweet = tokenized_tweet[0:(max_seq_length - 2)]
    tokens = []
    segment_ids = []
    tokens.append("[CLS]")
    segment_ids.append(0)
    for token in tokenized_tweet:
      tokens.append(token)
      segment_ids.append(0)
    tokens.append("[SEP]")
    segment_ids.append(0)

    input_ids = tokenizer.convert_tokens_to_ids(tokens)

    stats['avg_len'] += len(input_ids)/stats['len_examples']
    stats['seq_len_dist'][len(input_ids)] += 1
    if len(input_ids) > stats['max_len']:
      stats['max_len'] = len(input_ids)

    input_mask = [1] * len(input_ids)

    while len(input_ids) < max_seq_length:
      input_ids.append(0)
      input_mask.append(0)
      segment_ids.append(0)
    
    assert len(input_ids) == max_seq_length
    assert len(input_mask) == max_seq_length
    assert len(segment_ids) == max_seq_length

    stats[example.label.lower()] += 1

    if example_index < 20:
      tf.logging.info("*** Example ***")
      tf.logging.info("guid: {}".format(example.guid))
      tf.logging.info("tokens: {}".format(" ".join(tokens)))
      tf.logging.info("input_ids: {}".format(" ".join([str(x) for x in input_ids])))
      tf.logging.info("input_mask: {}".format(" ".join([str(x) for x in input_mask])))
      tf.logging.info("segment_ids: {}".format(" ".join([str(x) for x in segment_ids])))
      tf.logging.info("label: %s (id = %d)" % (example.label, example.label_id))
    

    feature = InputFeatures(
        input_ids=input_ids,
        input_mask=input_mask,
        segment_ids=segment_ids,
        label_id=example.label_id,
        is_real_example=True
    )
    return feature


  def _file_based_convert_examples_to_features(self,
      examples, max_seq_length, tokenizer, output_file):
    
    writer = tf.python_io.TFRecordWriter(output_file)

    stats = {'avg_len': 0, 'len_examples': len(examples), 'max_len': 0,
            'seq_len_dist': Counter()}
    
    for l in self.labels:
      stats[l.lower()] = 0

    for (ex_index, example) in enumerate(examples):
      if ex_index % 10000 == 0:
        tf.logging.info("Writing example %d of %d" % (ex_index, len(examples)))

      feature = self._convert_single_example(ex_index, example, stats, max_seq_length, 
                                      tokenizer)
      
      def _int64_feature(values):
        f = tf.train.Feature(int64_list=tf.train.Int64List(value=list(values)))
        return f
      
      def _float_feature(values):
        f = tf.train.Feature(float_list=tf.train.FloatList(value=list(values)))
        return f
      
      features = OrderedDict()
      features["input_ids"] = _int64_feature(feature.input_ids)
      features["input_mask"] = _int64_feature(feature.input_mask)
      features["segment_ids"] = _int64_feature(feature.segment_ids)
      features["label_ids"] = _int64_feature([feature.label_id])
      features["is_real_example"] = _int64_feature(
          [int(feature.is_real_example)])
      tf_example = tf.train.Example(features=tf.train.Features(feature=features))
      writer.write(tf_example.SerializeToString())
    writer.close()
    tf.logging.info('Average sequence length: {}'.format(stats['avg_len']))
    tf.logging.info('Number of examples: {}'.format(stats['len_examples']))
    tf.logging.info('Max length: {}'.format(stats['max_len']))
    tf.logging.info("Label distribution: " + ", ".join([l.upper() + ": {}" for l in self.labels]).format(*[stats[l.lower()] for l in self.labels]))
    return stats


class DavidsonExample(Example):
  """A single training/test example for the Davidson dataset.
  
  """

  def __init__(self, guid, tweet, label_id):
    self.labels = ['HATE', 'OFF', 'NONE']
    super().__init__(guid, tweet, label_id)

  def _set_label(self, label_id):
    return self.labels[label_id]
  
  def _set_label_id(self, label):
    try:
      return self.labels.index(label)
    except ValueError:
      raise ValueError("Label must be in ['HATE', 'OFF', 'NONE']")


class DavidsonProcessor(Processor):
  """Processor for Davidson Dataset"""
  def __init__(self, do_lower_case=True, dev_fraction=0.2, test_fraction=0.2, random_state=42, keep_df=True):
    super().__init__('davidson', do_lower_case, dev_fraction, test_fraction, random_state, keep_df)

  def _read_df(self):
    tf.logging.info("Reading dataframe")
    return pd.read_pickle('/content/drive/My Drive/masters_thesis/data/davidson/data/labeled_data.p')

  def get_labels(self):
    return ['HATE', 'OFF', 'NONE']

  def _get_example_from_row(self, row):
    return DavidsonExample(row.name, row.tweet, row['class'])


class FountaExample(Example):
  def __init__(self, guid, tweet, label_id):
    self.labels = ['abusive', 'hateful', 'normal', 'spam']
    super().__init__(guid, tweet, label_id)


  def _set_label(self, label_id):
    return self.labels[label_id]
  
  def _set_label_id(self, label):
    try:
      return self.labels.index(label)
    except ValueError:
      raise ValueError("Label must be in ['abusive', 'hateful', 'normal', 'spam']")


class FountaProcessor(Processor):
  def __init__(self, do_lower_case=True, dev_fraction=0.2, test_fraction=0.2, random_state=42, keep_df=True):
    super().__init__('founta', do_lower_case, dev_fraction, test_fraction, random_state, keep_df)
  
  def _get_example_from_row(self, row):
    return FountaExample(row.tweet_id, row.text, row.maj_label)
  
  def _read_df(self):
    print("Loading dataframe...")
    df = pd.read_pickle("/content/drive/My Drive/prosjektoppgave/data/founta/founta.pkl")
    print("Dataframe loaded.")
    return df

  def get_labels(self):
    return ['abusive', 'hateful', 'normal', 'spam']
    