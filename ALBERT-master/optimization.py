# coding=utf-8
# Copyright 2018 The Google AI Team Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# Lint as: python2, python3
"""Functions and classes related to optimization (weight updates)."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import re
import lamb_optimizer
import six
from six.moves import zip
import tensorflow.compat.v1 as tf
from tensorflow.contrib import tpu as contrib_tpu
from adamw import AdamWeightDecayOptimizer
from gradient_accumulation import use_gradient_accumulation


def create_optimizer(loss, init_lr, num_train_steps, num_warmup_steps, use_tpu,
                     optimizer="adamw", poly_power=1.0, start_warmup_step=0, gradient_accumulation_multiplier=1, do_update=None):
  """Creates an optimizer training op."""
  global_step = tf.train.get_or_create_global_step()

  use_accumulation = gradient_accumulation_multiplier > 1 and do_update is not None

  learning_rate = tf.constant(value=init_lr, shape=[], dtype=tf.float32)

  # Implements linear decay of the learning rate.
  learning_rate = tf.train.polynomial_decay(
      learning_rate,
      global_step,
      num_train_steps,
      end_learning_rate=0.0,
      power=poly_power,
      cycle=False)

  # Implements linear warmup. I.e., if global_step - start_warmup_step <
  # num_warmup_steps, the learning rate will be
  # `(global_step - start_warmup_step)/num_warmup_steps * init_lr`.
  if num_warmup_steps:
    tf.logging.info("++++++ warmup starts at step " + str(start_warmup_step)
                    + ", for " + str(num_warmup_steps) + " steps ++++++")
    global_steps_int = tf.cast(global_step, tf.int32)
    start_warm_int = tf.constant(start_warmup_step, dtype=tf.int32)
    global_steps_int = global_steps_int - start_warm_int
    warmup_steps_int = tf.constant(num_warmup_steps, dtype=tf.int32)

    global_steps_float = tf.cast(global_steps_int, tf.float32)
    warmup_steps_float = tf.cast(warmup_steps_int, tf.float32)

    warmup_percent_done = global_steps_float / warmup_steps_float
    warmup_learning_rate = init_lr * warmup_percent_done

    is_warmup = tf.cast(global_steps_int < warmup_steps_int, tf.float32)
    learning_rate = (
        (1.0 - is_warmup) * learning_rate + is_warmup * warmup_learning_rate)

  # It is OK that you use this optimizer for finetuning, since this
  # is how the model was trained (note that the Adam m/v variables are NOT
  # loaded from init_checkpoint.)
  # It is OK to use AdamW in the finetuning even the model is trained by LAMB.
  # As report in the Bert pulic github, the learning rate for SQuAD 1.1 finetune
  # is 3e-5, 4e-5 or 5e-5. For LAMB, the users can use 3e-4, 4e-4,or 5e-4 for a
  # batch size of 64 in the finetune.
  if optimizer == "adamw":
    tf.logging.info("using adamw")
    optimizer = AdamWeightDecayOptimizer(
        learning_rate=learning_rate,
        weight_decay_rate=0.01,
        beta_1=0.9,
        beta_2=0.999,
        epsilon=1e-6,
        exclude_from_weight_decay=["LayerNorm", "layer_norm", "bias"])
  elif optimizer == "lamb":
    tf.logging.info("using lamb")
    optimizer = lamb_optimizer.LAMBOptimizer(
        learning_rate=learning_rate,
        weight_decay_rate=0.01,
        beta_1=0.9,
        beta_2=0.999,
        epsilon=1e-6,
        exclude_from_weight_decay=["LayerNorm", "layer_norm", "bias"])
  else:
    raise ValueError("Not supported optimizer: ", optimizer)

  if use_tpu:
    optimizer = contrib_tpu.CrossShardOptimizer(optimizer)

  tvars = tf.trainable_variables()
  grads = tf.gradients(loss, tvars, aggregation_method=tf.AggregationMethod.EXPERIMENTAL_TREE)

  # This is how the model was pre-trained.
  (grads, _) = tf.clip_by_global_norm(grads, clip_norm=1.0)


  if use_accumulation:
    train_op = use_gradient_accumulation(grads, tvars, optimizer, gradient_accumulation_multiplier, global_step, do_update)
  else:
    train_op = optimizer.apply_gradients(
        list(zip(grads, tvars)), global_step=global_step)

      # Normally the global step update is done inside of `apply_gradients`.
    # However, neither `AdamWeightDecayOptimizer` nor `LAMBOptimizer` do this.
    # But if you use a different optimizer, you should probably take this line
    # out.
    new_global_step = global_step + 1
    train_op = tf.group(train_op, [global_step.assign(new_global_step)])

  
  return train_op
