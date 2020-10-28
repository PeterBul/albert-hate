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
import six
import tensorflow.compat.v1 as tf

# pylint: disable=g-direct-tensorflow-import
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import linalg_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.training import optimizer
from tensorflow.python.ops import state_ops
from tensorflow.python.ops import resource_variable_ops
# pylint: enable=g-direct-tensorflow-import


class LAMBOptimizer(tf.train.Optimizer):
  """LAMB (Layer-wise Adaptive Moments optimizer for Batch training)."""
  # A new optimizer that includes correct L2 weight decay, adaptive
  # element-wise updating, and layer-wise justification. The LAMB optimizer
  # was proposed by Yang You, Jing Li, Jonathan Hseu, Xiaodan Song,
  # James Demmel, and Cho-Jui Hsieh in a paper titled as Reducing BERT
  # Pre-Training Time from 3 Days to 76 Minutes (arxiv.org/abs/1904.00962)

  def __init__(self,
               learning_rate,
               weight_decay_rate=0.0,
               beta_1=0.9,
               beta_2=0.999,
               epsilon=1e-6,
               exclude_from_weight_decay=None,
               exclude_from_layer_adaptation=None,
               name="LAMBOptimizer"):
    """Constructs a LAMBOptimizer."""
    super(LAMBOptimizer, self).__init__(False, name)

    self.learning_rate = learning_rate
    self.weight_decay_rate = weight_decay_rate
    self.beta_1 = beta_1
    self.beta_2 = beta_2
    self.epsilon = epsilon
    self.exclude_from_weight_decay = exclude_from_weight_decay
    # exclude_from_layer_adaptation is set to exclude_from_weight_decay if the
    # arg is None.
    # TODO(jingli): validate if exclude_from_layer_adaptation is necessary.
    if exclude_from_layer_adaptation:
      self.exclude_from_layer_adaptation = exclude_from_layer_adaptation
    else:
      self.exclude_from_layer_adaptation = exclude_from_weight_decay
  
  

  def _create_slots(self, var_list):
    for v in var_list:
        self._zeros_slot(v, 'm', self._name)
    for v in var_list:
        self._zeros_slot(v, 'v', self._name)

  
  def _prepare(self):
    self.learning_rate_t = ops.convert_to_tensor(
        self.learning_rate, name='learning_rate')
    self.weight_decay_rate_t = ops.convert_to_tensor(
        self.weight_decay_rate, name='weight_decay_rate')
    self.beta_1_t = ops.convert_to_tensor(self.beta_1, name='beta_1')
    self.beta_2_t = ops.convert_to_tensor(self.beta_2, name='beta_2')
    self.epsilon_t = ops.convert_to_tensor(self.epsilon, name='epsilon')
  
    
  def _apply_dense(self, grad, var):
    return self._apply_lamb(grad, var)

  def _resource_apply_dense(self, grad, var):
    return self._apply_lamb(grad, var)


  def _apply_lamb(self, grad, var):
    var_name = self._get_variable_name(var.name)
    learning_rate_t = math_ops.cast(
      self.learning_rate_t, var.dtype.base_dtype)
    beta_1_t = math_ops.cast(self.beta_1_t, var.dtype.base_dtype)
    beta_2_t = math_ops.cast(self.beta_2_t, var.dtype.base_dtype)
    epsilon_t = math_ops.cast(self.epsilon_t, var.dtype.base_dtype)
    weight_decay_rate_t = math_ops.cast(
      self.weight_decay_rate_t, var.dtype.base_dtype)

    m = self.get_slot(var, 'm')
    v = self.get_slot(var, 'v')

    # Standard Adam update.
    next_m = (
      tf.multiply(beta_1_t, m) +
      tf.multiply(1.0 - beta_1_t, grad))
    next_v = (
      tf.multiply(beta_2_t, v) + tf.multiply(1.0 - beta_2_t,
                                            tf.square(grad)))
    
    update = next_m / (tf.sqrt(next_v) + epsilon_t)

    if self._do_use_weight_decay(var.name):
      update += weight_decay_rate_t * var
    
    ratio = 1.0

    if self._do_layer_adaptation(var_name):
      w_norm = linalg_ops.norm(var, ord=2)
      g_norm = linalg_ops.norm(update, ord=2)
      ratio = array_ops.where(math_ops.greater(w_norm, 0), array_ops.where(
          math_ops.greater(g_norm, 0), (w_norm / g_norm), 1.0), 1.0)

    update_with_lr = ratio * learning_rate_t * update

    next_param = var - update_with_lr
    
    return control_flow_ops.group(*[var.assign(next_param),
                                        m.assign(next_m),
                                        v.assign(next_v)])

    
  def _apply_sparse_shared(self, grad, var, indices, scatter_add):
    var_name = self._get_variable_name(var.name)
    learning_rate_t = math_ops.cast(
        self.learning_rate_t, var.dtype.base_dtype)
    beta_1_t = math_ops.cast(self.beta_1_t, var.dtype.base_dtype)
    beta_2_t = math_ops.cast(self.beta_2_t, var.dtype.base_dtype)
    epsilon_t = math_ops.cast(self.epsilon_t, var.dtype.base_dtype)
    weight_decay_rate_t = math_ops.cast(
        self.weight_decay_rate_t, var.dtype.base_dtype)

    m = self.get_slot(var, 'm')
    v = self.get_slot(var, 'v')

    m_t = state_ops.assign(m, m * beta_1_t,
                            use_locking=self._use_locking)

    m_scaled_g_values = grad * (1 - beta_1_t)

    with ops.control_dependencies([m_t]):
        m_t = scatter_add(m, indices, m_scaled_g_values)

    v_scaled_g_values = (grad * grad) * (1 - beta_2_t)
    v_t = state_ops.assign(v, v * beta_2_t, use_locking=self._use_locking)
    with ops.control_dependencies([v_t]):
        v_t = scatter_add(v, indices, v_scaled_g_values)

    update = m_t / (math_ops.sqrt(v_t) + epsilon_t)

    if self._do_use_weight_decay(var.name):
        update += weight_decay_rate_t * var


    ratio = 1.0
    if self._do_layer_adaptation(var_name):
      w_norm = linalg_ops.norm(var, ord=2)
      g_norm = linalg_ops.norm(update, ord=2)
      ratio = array_ops.where(math_ops.greater(w_norm, 0), array_ops.where(
          math_ops.greater(g_norm, 0), (w_norm / g_norm), 1.0), 1.0)

    update_with_lr = ratio * learning_rate_t * update

    var_update = state_ops.assign_sub(var,
                                      update_with_lr,
                                      use_locking=self._use_locking)
    return control_flow_ops.group(*[var_update, m_t, v_t])

  
  def _apply_sparse(self, grad, var):
    return self._apply_sparse_shared(
        grad.values, var, grad.indices,
        lambda x, i, v: state_ops.scatter_add(  # pylint: disable=g-long-lambda
            x, i, v, use_locking=self._use_locking))

  def _resource_scatter_add(self, x, i, v):
    with ops.control_dependencies(
        [resource_variable_ops.resource_scatter_add(
            x.handle, i, v)]):
        return x.value()

  def _resource_apply_sparse(self, grad, var, indices):
    return self._apply_sparse_shared(
        grad, var, indices, self._resource_scatter_add)
  


    
  def _do_use_weight_decay(self, param_name):
    """Whether to use L2 weight decay for `param_name`."""
    if not self.weight_decay_rate:
      return False
    if self.exclude_from_weight_decay:
      for r in self.exclude_from_weight_decay:
        if re.search(r, param_name) is not None:
          return False
    return True

  def _do_layer_adaptation(self, param_name):
    """Whether to do layer-wise learning rate adaptation for `param_name`."""
    if self.exclude_from_layer_adaptation:
      for r in self.exclude_from_layer_adaptation:
        if re.search(r, param_name) is not None:
          return False
    return True

  def _get_variable_name(self, param_name):
    """Get the variable name from the tensor name."""
    m = re.match("^(.*):\\d+$", six.ensure_str(param_name))
    if m is not None:
      param_name = m.group(1)
    return param_name

