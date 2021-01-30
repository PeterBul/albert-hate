from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
from args import parser


args = parser.parse_args()

PATH_DATASET= 'data' + os.sep + 'tfrecords'

SOLID_CONVERTED = PATH_DATASET + os.sep + args.dataset + os.sep + 'train-' + str(args.sequence_length) + '.tfrecords'
OLID_CONVERTED = PATH_DATASET + os.sep + args.dataset + os.sep + 'dev-' + str(args.sequence_length) + '.tfrecords'
SOLID_CONVERTED_TEST = PATH_DATASET + os.sep + args.dataset + os.sep + 'test-' + str(args.sequence_length) + '.tfrecords'