# experiments.py
"""
Configuration and implementation of different Hopfield Network experiments.
"""

import numpy as np
from typing import Dict, List, Tuple


class ExperimentConfig:
  """Base configuration for experiments"""

  def __init__(self, name: str):
    self.base_config = {
      'name': name,
      'use_hebb': True,
      'use_oja': True,
      'learning_rate_oja': 0.01,
      'epochs_oja': 100,
      'corruption_level': 0.3,
      'async_update': False,
      'threshold': 0,
      'show_plots': False,
      'show_weights': False,
      'save_plots': True,
    }


class LearningRulesComparison(ExperimentConfig):
  """Compare effectiveness of Hebb and Oja learning rules"""

  def __init__(self):
    super().__init__('learning_rules_comparison')
    self.config = self.base_config.copy()
    self.config.update({
      'corruption_levels': [0.1, 0.2, 0.3, 0.4],
    })


class PatternStability(ExperimentConfig):
  """Analyze pattern stability and capacity"""

  def __init__(self):
    super().__init__('pattern_stability')
    self.config = self.base_config.copy()
    self.config.update({
      'corruption_level': 0,  # No corruption for stability test
      'test_stability': True,
    })


class SizeComparison(ExperimentConfig):
  """Compare 25x25 vs 25x50 patterns"""

  def __init__(self):
    super().__init__('size_comparison')
    self.config = self.base_config.copy()
    self.config.update({
      'target_files': ['large-25x25.csv', 'large-25x50.csv'],
      'corruption_levels': [0.1, 0.2, 0.3],
    })


class RecoveryAnalysis(ExperimentConfig):
  """Analyze recovery effectiveness by set size"""

  def __init__(self):
    super().__init__('recovery_analysis')
    self.config = self.base_config.copy()
    self.config.update({
      'corruption_levels': [0.1, 0.2, 0.3, 0.4, 0.5],
      'analyze_set_size': True,
    })


class PatternSimilarity(ExperimentConfig):
  """Compare similar vs diverse pattern sets"""

  def __init__(self):
    super().__init__('pattern_similarity')
    self.config = self.base_config.copy()
    self.config.update({
      'analyze_similarity': True,
    })


class RandomInput(ExperimentConfig):
  """Test network response to random inputs"""

  def __init__(self):
    super().__init__('random_input')
    self.config = self.base_config.copy()
    self.config.update({
      'num_random_inputs': 10,
      'test_random': True,
    })


class MaxStable5x5(ExperimentConfig):
  """Find maximum stable 5x5 pattern set"""

  def __init__(self):
    super().__init__('max_stable_5x5')
    self.config = self.base_config.copy()
    self.config.update({
      'pattern_shape': (5, 5),
      'max_patterns': 100,
      'stability_threshold': 0.99,
      'find_max_stable': True,
    })


class OscillationDetection(ExperimentConfig):
  """Detect oscillating patterns"""

  def __init__(self):
    super().__init__('oscillation')
    self.config = self.base_config.copy()
    self.config.update({
      'max_iterations': 100,
      'detect_oscillation': True,
    })


class LargeBitmap(ExperimentConfig):
  """Process large bitmap patterns"""

  def __init__(self):
    super().__init__('large_bitmap')
    self.config = self.base_config.copy()
    self.config.update({
      'min_size': (200, 300),
      'process_large': True,
    })


# Experiment-specific functions
def test_pattern_stability(model, patterns: np.ndarray) -> Tuple[
  List[bool], float]:
  """
  Test if patterns are stable states.
  Returns: (stable_flags, stability_ratio)
  """
  stable_flags = []
  for pattern in patterns:
    predicted = model.predict([pattern], threshold=0, asyn=False)[0]
    stable_flags.append(np.array_equal(pattern, predicted))
  stability_ratio = sum(stable_flags) / len(stable_flags)
  return stable_flags, stability_ratio


def analyze_pattern_similarity(patterns: np.ndarray) -> Dict[str, float]:
  """
  Calculate similarity between patterns in a set.
  Returns: Dictionary with similarity metrics
  """
  n_patterns = len(patterns)
  similarities = []

  for i in range(n_patterns):
    for j in range(i + 1, n_patterns):
      sim = np.sum(patterns[i] == patterns[j]) / len(patterns[i])
      similarities.append(sim)

  return {
    'mean_similarity': np.mean(similarities),
    'std_similarity': np.std(similarities),
    'min_similarity': np.min(similarities),
    'max_similarity': np.max(similarities),
  }


def detect_oscillation(model, pattern: np.ndarray, max_iterations: int = 100) -> \
Tuple[bool, List[np.ndarray]]:
  """
  Check if pattern leads to oscillation.
  Returns: (oscillates, state_history)
  """
  state_history = [pattern]
  seen_states = {tuple(pattern)}

  for _ in range(max_iterations):
    next_state = model.predict([state_history[-1]], threshold=0, asyn=False)[0]
    state_tuple = tuple(next_state)

    if state_tuple in seen_states:
      cycle_start = state_history.index(next_state)
      return True, state_history[cycle_start:]

    state_history.append(next_state)
    seen_states.add(state_tuple)

  return False, state_history


# Get all available experiments
def get_experiments() -> Dict[str, ExperimentConfig]:
  """Return dictionary of all available experiments"""
  return {
    'learning_rules': LearningRulesComparison(),
    'stability': PatternStability(),
    'size_comparison': SizeComparison(),
    'recovery': RecoveryAnalysis(),
    'similarity': PatternSimilarity(),
    'random_input': RandomInput(),
    'max_stable_5x5': MaxStable5x5(),
    'oscillation': OscillationDetection(),
    'large_bitmap': LargeBitmap(),
  }
