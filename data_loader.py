import numpy as np
import re
import os


def load_csv_patterns(filename):
  """
  Load patterns from a single CSV file where each row is a flattened pattern.
  The shape is determined from the filename (e.g., '25x25' means each pattern is 25x25).

  Args:
      filename: Name of the CSV file (e.g., 'large-25x25.csv')

  Returns:
      patterns: numpy array of shape (num_patterns, pattern_size)
      pattern_shape: tuple of (height, width)
  """
  # Extract shape from filename (e.g., 'large-25x25.csv' -> (25, 25))
  match = re.search(r'(\d+)x(\d+)', filename)
  if not match:
    raise ValueError(f"Cannot extract shape from filename: {filename}")

  height, width = map(int, match.groups())
  pattern_size = height * width

  # Load the CSV file - each row is a pattern
  data = np.loadtxt(os.path.join('input_images', filename), delimiter=',')
  num_patterns = data.shape[0]

  print(f"\nLoaded {filename}:")
  print(f"- Number of patterns: {num_patterns}")
  print(f"- Pattern shape: {height}x{width}")

  return data, (height, width)


def get_csv_files(input_dir='input_images'):
    """Get sorted list of CSV files from directory."""
    try:
      files = sorted(f for f in os.listdir(input_dir) if f.endswith('.csv'))
      print(f"Found {len(files)} CSV files to process:",
            *[f"- {f}" for f in files], sep='\n')
      return files
    except (FileNotFoundError, PermissionError) as e:
      print(f"Error accessing directory '{input_dir}': {e}")
      return []


def ensure_experiment_dir(experiment_name):
  """Create output directory structure for experiment"""
  base_dir = 'output'
  experiment_dir = os.path.join(base_dir, experiment_name)

  # Create directories for each learning rule
  dirs = {}
  for rule in ['hebb', 'oja']:
    rule_dir = os.path.join(experiment_dir, rule)
    dirs[rule] = {
      'plots': os.path.join(rule_dir, 'plots'),
      'weights': os.path.join(rule_dir, 'weights'),
      'results': os.path.join(rule_dir, 'results')
    }
    for directory in dirs[rule].values():
      if not os.path.exists(directory):
        os.makedirs(directory)

  return dirs
