import unittest
import preprocessing
import pandas as pd

class TestPreprocessing(unittest.TestCase):

  def test_get_train_dev_test_dataframes(self):
    processor = preprocessing.FountaConvProcessor('data')
    train, dev, test = processor.get_train_dev_test_dataframes()
    self.assertIsInstance(train, pd.DataFrame)
    self.assertIsInstance(dev, pd.DataFrame)
    self.assertIsInstance(test, pd.DataFrame)

  

if __name__ == "__main__":
  unittest.main()