from tensorflow import

def apply_gradients(self, grads_and_vars, global_step=None, name=None):
  """Apply gradients to variables.
  This is the second part of `minimize()`. It returns an `Operation` that
  applies gradients.
  Args:
    grads_and_vars: List of (gradient, variable) pairs as returned by
      `compute_gradients()`.
    global_step: Optional `Variable` to increment by one after the
      variables have been updated.
    name: Optional name for the returned operation.  Default to the
      name passed to the `Optimizer` constructor.
  Returns:
    An `Operation` that applies the specified gradients. If `global_step`
    was not None, that operation also increments `global_step`.
  Raises:
    TypeError: If `grads_and_vars` is malformed.
    ValueError: If none of the variables have gradients.
    RuntimeError: If you should use `_distributed_apply()` instead.
  """
  # This is a default implementation of apply_gradients() that can be shared
  # by most optimizers.  It relies on the subclass implementing the following
  # methods: _create_slots(), _prepare(), _apply_dense(), and _apply_sparse().

  # TODO(isaprykin): Get rid of `has_strategy()` check by
  # always calling _distributed_apply(), using the default distribution
  # as needed.
  if distribute_ctx.has_strategy():
    # Handle DistributionStrategy case.
    if distribute_ctx.in_cross_replica_context():
      raise RuntimeError("Use `_distributed_apply()` instead of "
                          "`apply_gradients()` in a cross-replica context.")

    grads_and_vars = get_filtered_grad_fn(lambda: grads_and_vars)()
    return distribute_ctx.get_replica_context().merge_call(
        self._distributed_apply, args=(grads_and_vars, global_step, name))

  # No DistributionStrategy case.
  grads_and_vars = tuple(grads_and_vars)  # Make sure repeat iteration works.
  if not grads_and_vars:
    raise ValueError("No variables provided.")
  converted_grads_and_vars = []
  for g, v in grads_and_vars:
    if g is not None:
      try:
        # Convert the grad to Tensor or IndexedSlices if necessary.
        g = ops.convert_to_tensor_or_indexed_slices(g)
      except TypeError:
        raise TypeError(
            "Gradient must be convertible to a Tensor"
            " or IndexedSlices, or None: %s" % g)
      if not isinstance(g, (ops.Tensor, ops.IndexedSlices)):
        raise TypeError(
            "Gradient must be a Tensor, IndexedSlices, or None: %s" % g)
    p = _get_processor(v)
    converted_grads_and_vars.append((g, v, p))

  converted_grads_and_vars = tuple(converted_grads_and_vars)
  var_list = [v for g, v, _ in converted_grads_and_vars if g is not None]
  if not var_list:
    raise ValueError("No gradients provided for any variable: %s." %
                      ([str(v) for _, v, _ in converted_grads_and_vars],))
  with ops.init_scope():
    self._create_slots(var_list)
  update_ops = []
  with ops.name_scope(name, self._name, skip_on_eager=False) as name:
    self._prepare()
    for grad, var, processor in converted_grads_and_vars:
      if grad is None:
        continue
      # We colocate all ops created in _apply_dense or _apply_sparse
      # on the same device as the variable.
      # TODO(apassos): figure out how to get the variable name here.
      if (context.executing_eagerly() or
          resource_variable_ops.is_resource_variable(var)
          and not var._in_graph_mode):  # pylint: disable=protected-access
        scope_name = ""
      else:
        scope_name = var.op.name
      with ops.name_scope(
          "update_" + scope_name,
          skip_on_eager=False), ops.colocate_with(var):
        update_ops.append(processor.update_op(self, grad))
    if global_step is None:
      apply_updates = self._finish(update_ops, name)
    else:
      with ops.control_dependencies([self._finish(update_ops, "update")]):
        with ops.colocate_with(global_step):
          if isinstance(
              global_step, resource_variable_ops.BaseResourceVariable):
            # TODO(apassos): the implicit read in assign_add is slow; consider
            # making it less so.
            apply_updates = resource_variable_ops.assign_add_variable_op(
                global_step.handle,
                ops.convert_to_tensor(1, dtype=global_step.dtype),
                name=name)
          else:
            apply_updates = state_ops.assign_add(global_step, 1, name=name)

    if not context.executing_eagerly():
      if isinstance(apply_updates, ops.Tensor):
        apply_updates = apply_updates.op
      train_op = ops.get_collection_ref(ops.GraphKeys.TRAIN_OP)
      if apply_updates not in train_op:
        train_op.append(apply_updates)

    return apply_updates



def _distributed_apply(self,
                         distribution,
                         grads_and_vars,
                         global_step=None,
                         name=None):
  """A version of `apply_gradients` for cross-replica context.
  This is a version of `apply_gradients()` for when you are using a
  `DistributionStrategy` and are in a cross-replica context. If in a
  replica context, use `apply_gradients()` as normal.
  Args:
    distribution: A `DistributionStrategy` object.
    grads_and_vars: List of (gradient, variable) pairs as returned by
      `compute_gradients()`, and then aggregated across replicas.
    global_step: Optional (mirrored) `Variable` to increment by one
      after the variables have been updated.
    name: Optional name for the returned operation.  Default to the
      name passed to the `Optimizer` constructor.
  Returns:
    An `Operation` that applies the specified gradients across all
    replicas. If `global_step` was not None, that operation also
    increments `global_step`
  """
  reduced_grads = distribution.extended.batch_reduce_to(
      ds_reduce_util.ReduceOp.SUM, grads_and_vars)
  var_list = [v for _, v in grads_and_vars]
  grads_and_vars = zip(reduced_grads, var_list)

  # Note that this is called in a cross-replica context.
  with ops.init_scope():
    self._create_slots(var_list)

  def update(v, g):
    """Apply gradients to a replica variable."""
    assert v is not None

    try:
      # Convert the grad to Tensor or IndexedSlices if necessary.
      g = ops.convert_to_tensor_or_indexed_slices(g)
    except TypeError:
      raise TypeError("Gradient must be convertible to a Tensor"
                      " or IndexedSlices, or None: %s" % g)
    if not isinstance(g, (ops.Tensor, ops.IndexedSlices)):
      raise TypeError(
          "Gradient must be a Tensor, IndexedSlices, or None: %s" % g)
    p = _get_processor(v)

    if context.executing_eagerly() or (
        resource_variable_ops.is_resource_variable(v) and
        not v._in_graph_mode):  # pylint: disable=protected-access
      scope_name = v.name.split(":")[0]
    else:
      scope_name = v.op.name

    # device_policy is set because non-mirrored tensors will be read in
    # `update_op`. `_resource_apply_dense`, `lr_t`, `beta1_t` and `beta2_t`
    # is an example.
    with ops.name_scope("update_" + scope_name):
      return p.update_op(self, g)

  with ops.name_scope(name, self._name) as name:
    self._prepare()

    update_ops = [
        op
        for grad, var in grads_and_vars
        for op in distribution.extended.update(
            var, update, args=(grad,), group=False)
    ]

    def finish(self, update_ops):
      return self._finish(update_ops, "update")

    non_slot_devices = distribution.extended.non_slot_devices(var_list)
    finish_updates = distribution.extended.update_non_slot(
        non_slot_devices, finish, args=(self, update_ops), group=False)
    if global_step is None:
      apply_updates = distribution.group(finish_updates, name=name)
    else:
      with ops.control_dependencies(finish_updates):
        apply_updates = distribution.extended.update(
            global_step, state_ops.assign_add, args=(1,),
            kwargs={"name": name})

    if not context.executing_eagerly():
      if isinstance(apply_updates, ops.Tensor):
        apply_updates = apply_updates.op
      train_op = ops.get_collection_ref(ops.GraphKeys.TRAIN_OP)
      if apply_updates not in train_op:
        train_op.append(apply_updates)

    return apply_updates