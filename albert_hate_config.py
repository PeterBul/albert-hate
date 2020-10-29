import utils                                                          #pylint: disable=import-error
import six
import json
import copy
import tensorflow as tf


class AlbertHateConfig(object):
    def __init__(self,
                    linear_layers=0,
                    model_dir=None,
                    model_size='base',
                    num_labels=2,
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