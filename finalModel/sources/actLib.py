#Copyright (C) 2017-2019 Guenter Klambauer
#License according to the origin of this source: https://github.com/bioinf-jku/SNNs/blob/master/selu.py

from __future__ import absolute_import, division, print_function
import numbers
import numpy as np
import tensorflow as tf
from tensorflow.contrib import layers
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_shape
from tensorflow.python.framework import tensor_util
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import random_ops
from tensorflow.python.ops import array_ops
from tensorflow.python.layers import utils

def selu(x):
  alpha = 1.6732632423543772848170429916717
  scale = 1.0507009873554804934193349852946
  return scale*tf.where(x>=0.0, x, alpha*tf.nn.elu(x))

def dropout_selu(x, rate, alpha= -1.7580993408473766, fixedPointMean=0.0, fixedPointVar=1.0, noise_shape=None, seed=None, name=None, training=False):
  def dropout_selu_impl(x, rate, alpha, noise_shape, seed, name):
    keep_prob = 1.0 - rate
    x = ops.convert_to_tensor(x, name="x")
    if isinstance(keep_prob, numbers.Real) and not 0 < keep_prob <= 1:
      raise ValueError("keep_prob must be a scalar tensor or a float in the range (0, 1], got %g" % keep_prob)
    keep_prob = ops.convert_to_tensor(keep_prob, dtype=x.dtype, name="keep_prob")
    keep_prob.get_shape().assert_is_compatible_with(tensor_shape.scalar())

    alpha = ops.convert_to_tensor(alpha, dtype=x.dtype, name="alpha")
    keep_prob.get_shape().assert_is_compatible_with(tensor_shape.scalar())

    if tensor_util.constant_value(keep_prob) == 1:
      return x

    noise_shape = noise_shape if noise_shape is not None else array_ops.shape(x)
    random_tensor = keep_prob
    random_tensor += random_ops.random_uniform(noise_shape, seed=seed, dtype=x.dtype)
    binary_tensor = math_ops.floor(random_tensor)
    ret = x * binary_tensor + alpha * (1-binary_tensor)

    a = tf.sqrt(fixedPointVar / (keep_prob *((1-keep_prob) * tf.pow(alpha-fixedPointMean,2) + fixedPointVar)))

    b = fixedPointMean - a * (keep_prob * fixedPointMean + (1 - keep_prob) * alpha)
    ret = a * ret + b
    ret.set_shape(x.get_shape())
    return ret

  with ops.name_scope(name, "dropout", [x]) as name:
    return utils.smart_cond(training,
      lambda: dropout_selu_impl(x, rate, alpha, noise_shape, seed, name),
      lambda: array_ops.identity(x))

def dropout_stableVariance(x, rate, noise_shape=None, seed=None, name=None, training=False):
  def dropout_stableVariance_impl(x, rate, noise_shape, seed, name):
    keep_prob = 1.0 - rate
    x = ops.convert_to_tensor(x, name="x")
    if isinstance(keep_prob, numbers.Real) and not 0 < keep_prob <= 1:
      raise ValueError("keep_prob must be a scalar tensor or a float in the range (0, 1], got %g" % keep_prob)
    keep_prob = ops.convert_to_tensor(keep_prob, dtype=x.dtype, name="keep_prob")
    keep_prob.get_shape().assert_is_compatible_with(tensor_shape.scalar())

    if tensor_util.constant_value(keep_prob) == 1:
      return x

    noise_shape = noise_shape if noise_shape is not None else array_ops.shape(x)
    random_tensor = keep_prob
    random_tensor += random_ops.random_uniform(noise_shape, seed=seed, dtype=x.dtype)
    binary_tensor = math_ops.floor(random_tensor)
    ret = math_ops.div(x, math_ops.sqrt(keep_prob)) * binary_tensor
    ret.set_shape(x.get_shape())
    return ret
  
  with ops.name_scope(name, "dropout", [x]) as name:
    return utils.smart_cond(training,
      lambda: dropout_stableVariance_impl(x, rate, noise_shape, seed, name),
      lambda: array_ops.identity(x))

def dropout_relu(x, rate, name=None, training=False):
  with ops.name_scope(name, "dropout", [x]) as name:
    return utils.smart_cond(training,
      lambda: tf.nn.dropout(x, 1.0-rate),
      lambda: array_ops.identity(x))
