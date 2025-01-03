# config.py

class ExperimentConfig:
  """Configuration for all experiments"""

  def __init__(self):
    self.config = {
      # Learning configuration
      'use_hebb': True,
      'use_oja': True,
      'learning_rate_oja': 0.001,
      'epochs_oja': 3,

      # Testing configuration
      'corruption_levels': [0.1, 0.2, 0.3, 0.4, 0.5],
      'async_update': False,
      'threshold': 0,

      # Visualization configuration
      'show_plots': False,
      'show_weights': False,
      'save_plots': True,
    }


def get_config():
  """Return experiment configuration"""
  return ExperimentConfig().config
