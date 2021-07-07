"""Tests for vtrace_ops."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# Dependency imports
import unittest

from absl.testing import parameterized
import numpy as np
import tensorflow as tf
import vtrace_ops

def _shaped_arange(*shape):
  """Runs np.arange, converts to float and reshapes."""
  return np.arange(np.prod(shape), dtype=np.float32).reshape(*shape)


def _softmax(logits):
  """Applies softmax non-linearity on inputs."""
  return np.exp(logits) / np.sum(np.exp(logits), axis=-1, keepdims=True)


def _ground_truth_calculation(discounts, log_rhos, rewards, values,
                              bootstrap_value, clip_rho_threshold,
                              clip_pg_rho_threshold):
  """Calculates the ground truth for V-trace in Python/Numpy."""
  vs = []
  seq_len = len(discounts)
  rhos = np.exp(log_rhos)
  cs = np.minimum(rhos, 1.0)
  clipped_rhos = rhos
  if clip_rho_threshold:
    clipped_rhos = np.minimum(rhos, clip_rho_threshold)
  clipped_pg_rhos = rhos
  if clip_pg_rho_threshold:
    clipped_pg_rhos = np.minimum(rhos, clip_pg_rho_threshold)

  # This is a very inefficient way to calculate the V-trace ground truth.
  # We calculate it this way because it is close to the mathematical notation of
  # V-trace.
  # v_s = V(x_s)
  #       + \sum^{T-1}_{t=s} \gamma^{t-s}
  #         * \prod_{i=s}^{t-1} c_i
  #         * \rho_t (r_t + \gamma V(x_{t+1}) - V(x_t))
  # Note that when we take the product over c_i, we write `s:t` as the notation
  # of the paper is inclusive of the `t-1`, but Python is exclusive.
  # Also note that np.prod([]) == 1.
  values_t_plus_1 = np.concatenate([values, bootstrap_value[None, :]], axis=0)
  for s in range(seq_len):
    v_s = np.copy(values[s])  # Very important copy.
    for t in range(s, seq_len):
      v_s += (
          np.prod(discounts[s:t], axis=0) * np.prod(cs[s:t - 1],
                                                    axis=0) * clipped_rhos[t] *
          (rewards[t] + discounts[t] * values_t_plus_1[t + 1] - values[t]))
    vs.append(v_s)
  vs = np.stack(vs, axis=0)
  pg_advantages = (
      clipped_pg_rhos * (rewards + discounts * np.concatenate(
          [vs[1:], bootstrap_value[None, :]], axis=0) - values))

  return vtrace_ops.VTraceReturns(vs=vs, pg_advantages=pg_advantages)

class VtraceTest(tf.test.TestCase, parameterized.TestCase):

  @parameterized.named_parameters(('Batch1', 1), ('Batch5', 5))
  def testVTrace(self, batch_size):
    """Tests V-trace against ground truth data calculated in python."""
    seq_len = 5

    values = {
        # Note that this is only for testing purposes using well-formed inputs.
        # In practice we'd be more careful about taking log() of arbitrary
        # quantities.
        'log_rhos':
            np.log((_shaped_arange(seq_len, batch_size)) / batch_size /
                   seq_len + 1),
        # T, B where B_i: [0.9 / (i+1)] * T
        'discounts':
            np.array([[0.9 / (b + 1)
                       for b in range(batch_size)]
                      for _ in range(seq_len)]),
        'rewards':
            _shaped_arange(seq_len, batch_size),
        'values':
            _shaped_arange(seq_len, batch_size) / batch_size,
        'bootstrap_value':
            _shaped_arange(batch_size) + 1.0,
        'clip_rho_threshold':
            3.7,
        'clip_pg_rho_threshold':
            2.2,
    }
    print("test  input values",values)

    output = vtrace_ops.vtrace_from_importance_weights(**values)

    with self.test_session() as session:
      output_v = session.run(output)

    ground_truth_v = _ground_truth_calculation(**values)
    for a, b in zip(ground_truth_v, output_v):
      self.assertAllClose(a, b)


def make_suite():
    suite = tf.test.TestSuite()
    #suite.addTest(suiteTest("testadd"))
    #suite.addTest(suiteTest("testsub"))
    suite.addTest((tf.test.makeSuite(VtraceTest)))
    return suite

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    runner = unittest.TextTestRunner()
    test_suite = make_suite()
    runner.run(test_suite)

