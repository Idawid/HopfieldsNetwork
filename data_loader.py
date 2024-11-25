import numpy as np
np.random.seed(1)
import re
import os

import pandas as pd


def load_data(filename):
  for directory in ['input_images', 'custom_input_images']:
    try:
      return np.loadtxt(os.path.join(directory, filename), delimiter=',')
    except FileNotFoundError:
      continue
  raise FileNotFoundError(f"File {filename} not found in any input directory")


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
  data = load_data(filename)
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


def generate_random_patterns(num_patterns=5, pattern_shape=(25, 25),
    pattern_type='binary', sparsity=0.5):
  """
  Generate random patterns for testing Hopfield networks.
  Args:
      num_patterns: Number of patterns to generate
      pattern_shape: Tuple of (height, width)
      pattern_type: 'binary' (-1/1) or 'sparse' (mostly -1s with some 1s)
      sparsity: For sparse patterns, proportion of 1s (default 0.5)
  """
  height, width = pattern_shape
  pattern_size = height * width

  if pattern_type == 'binary':
    # Generate random binary patterns (-1 or 1)
    patterns = np.random.choice([-1, 1], size=(num_patterns, pattern_size))

  elif pattern_type == 'sparse':
    # Generate sparse patterns
    patterns = np.full((num_patterns, pattern_size), -1)
    for i in range(num_patterns):
      # Randomly set some elements to 1 based on sparsity
      num_ones = int(pattern_size * sparsity)
      ones_indices = np.random.choice(
          pattern_size,
          size=num_ones,
          replace=False
      )
      patterns[i, ones_indices] = 1

  else:
    raise ValueError(f"Unknown pattern type: {pattern_type}")

  print(f"\nGenerated random patterns:")
  print(f"- Number of patterns: {num_patterns}")
  print(f"- Pattern shape: {height}x{width}")
  print(f"- Pattern type: {pattern_type}")
  if pattern_type == 'sparse':
    print(f"- Sparsity: {sparsity}")

  return patterns, pattern_shape


def generate_custom_input(shapes=[(25, 25, 5), (20, 20, 3), (10, 10, 4)],
      output_dir='custom_input_images'):
    """
    Generate and save random patterns as CSV files.
    Args:
        shapes: List of tuples (height, width, num_patterns)
        output_dir: Directory to save files
    """
    os.makedirs(output_dir, exist_ok=True)

    for height, width, num_patterns in shapes:
      # Generate patterns
      pattern_size = height * width
      patterns = np.random.choice([-1, 1], size=(num_patterns, pattern_size))

      # Create dynamic filename
      filename = f"random_{height}x{width}_{num_patterns}.csv"

      # Save as CSV
      pd.DataFrame(patterns).to_csv(
          os.path.join(output_dir, filename),
          index=False,
          header=False
      )


def get_custom_input(input_dir='custom_input_images'):
  """Get sorted list of custom CSV files from directory."""
  try:
    files = sorted(f for f in os.listdir(input_dir) if f.endswith('.csv'))
    print(f"Found {len(files)} custom CSV files to process:",
          *[f"- {f}" for f in files], sep='\n')
    return files
  except (FileNotFoundError, PermissionError) as e:
    print(f"Error accessing directory '{input_dir}': {e}")
    return []
