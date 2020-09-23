from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow.compat.v1 as tf
from tensorflow.contrib import tpu as contrib_tpu


def use_gradient_accumulation(grads, tvars, optimizer, gradient_accmulation_multiplier, global_step, do_update):


    # declare a temp variable for summation
    sum_gradient = [tf.get_variable(name="sum_grads" + str(i), shape=tv.shape,
                                    initializer=tf.zeros_initializer,
                                    trainable=False,
                                    dtype=tf.float32,
                                    collections=[tf.GraphKeys.LOCAL_VARIABLES], aggregation=tf.VariableAggregation.MEAN) for i, tv in enumerate(tvars)]
    sum_ops = []
    unused_variable_in_batch = []

    # gradient accumulation
    for i, gv in enumerate(grads):
        if gv is not None:
            sum_ops.append(sum_gradient[i].assign_add(gv, name="accumulate_gradient"))
        else:
            unused_variable_in_batch.append(sum_gradient[i])
            sum_gradient[i] = None

    # NOTE : calling .assign_add does NOTHING in estimator, must wrap them all and handle them via train_ops

    def apply_accumulated_gradients(sums):
        # normalize gradient
        normalize_ops = []
        for i, g in enumerate(sums):
            if g is not None:
                # gradient_accmulation_multiplier is the number of steps to take before applying the gradients
                normalize_ops.append(sums[i].assign(tf.multiply(g, 1 / gradient_accmulation_multiplier)))
                # assign to make sure it still is a variable, or else it will become a Tensor
        with tf.control_dependencies(normalize_ops):
            minimize_op = optimizer.apply_gradients(zip(sums, tvars), global_step=global_step)
        return tf.group(minimize_op, *normalize_ops, name="apply_accumulated_gradients")

    train_op = tf.cond(tf.math.equal(global_step % gradient_accmulation_multiplier, 0),
                    lambda: apply_accumulated_gradients(sum_gradient),
                    tf.no_op)
                    #lambda: optimizer.apply_gradients(zip([None for _ in grads], tvars), global_step=global_step))

    # reset accumulation when necessary
    def reset():
        counter = 0
        for i, s in enumerate(sum_gradient):
            if s is None:
                # restore reference from None to the original variable
                sum_gradient[i] = unused_variable_in_batch[counter]
                counter += 1
        return tf.group([s.assign(tf.zeros_like(s)) for s in sum_gradient])

    with tf.control_dependencies([train_op]):
        reset_ops = tf.cond(tf.math.equal(do_update, 1.),
                            reset,
                            tf.no_op)
    # the 2 branches must have identical structure, [op1, op2, ...] || no_op cannot be valid cond branch.
    # tf.group to convert all resets into 1 op and match with no_op: tf.group() || np_op

    new_global_step = global_step + 1
    train_op = tf.group(*sum_ops, [train_op, global_step.assign(new_global_step), reset_ops])

    return train_op


class GradientAccumulationHook(tf.train.SessionRunHook):
    """
    Puts a certain tf.Variable to 1 once every certain steps.
    """


    def __init__(self, frequency, variable):
        self._step = 0
        self._flag = 0.
        self._freq = frequency
        self._input_placeholder = tf.placeholder(tf.float32)
        self.assign_op = variable.assign(self._input_placeholder)

    def begin(self):
        # a hook can modify graph at begin(), after this the graph will be finalized
        self._step = tf.train.get_global_step()

    def before_run(self, run_context):
        step = run_context.session.run(self._step)  # evaluate tensor to get a step number
        self._flag = 1. if step % self._freq == 0 and step != 0 else 0.
        run_context.session.run(self.assign_op, feed_dict={self._input_placeholder: self._flag})