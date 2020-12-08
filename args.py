from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import utils

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
parser.add_argument('--num_labels', default=None)

# Control arguments
parser.add_argument('--config_path', type=str, default=None, help="Path to model config for albert hate")
parser.add_argument('--init_checkpoint', type=str, default=None, help="Checkpoint to train from")
parser.add_argument('--wandb_run_path', type=str, default=None)
parser.add_argument('--model_dir', type=str, help='The name of the folder to save the model')
parser.add_argument('--predict', type=utils.str2bool, const=True, default=False, nargs='?')
parser.add_argument('--debug', type=utils.str2bool, const=True, default=False, nargs='?', help='Set flag to debug')
parser.add_argument('--test', type=utils.str2bool, const=True, default=False, nargs='?', help='Set flag to test')
parser.add_argument('--dataset', type=str, default='solid', help='The name of the dataset to be used. For now,  davidson or solid')
parser.add_argument('--task', type=str, default='a', help='Task a, b or c for solid/olid dataset')
parser.add_argument('--tree_predict', type=utils.str2bool, const=True, default=False, nargs='?')
parser.add_argument('--ensamble', type=utils.str2bool, const=True, default=False, nargs='?')
parser.add_argument('--dry_run', type=utils.str2bool, const=True, default=False, nargs='?')
parser.add_argument('--do_eval', type=utils.str2bool, const=True, default=False, nargs='?')

# Arguments that aren't very important
parser.add_argument('--epochs', type=int, default=-1, help='Number of epochs to train. If not set, iterations are used instead.')
parser.add_argument('--dataset_length', type=int, default=-1, help='Length of dataset. Is needed to caluclate iterations if epochs is used.')
