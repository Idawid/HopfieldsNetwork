import numpy as np
np.random.seed(1)
import re
import os
from skimage import io
from skimage.color import rgb2gray
from skimage.transform import resize

import pandas as pd


def load_data(filename):
  for directory in ['input_images', 'custom_input_images']:
    try:
      data = np.loadtxt(os.path.join(directory, filename), delimiter=',')
      if len(data.shape) == 1:  # If 1D array (m,)
        data = data.reshape(1, -1)  # Convert to 2D array (1,m)
      return data
    except FileNotFoundError:
      continue
  raise FileNotFoundError(f"File {filename} not found in any input directory")


def load_csv_patterns(filename):
  """Load and reshape patterns to correct 2D shape"""
  match = re.search(r'(\d+)x(\d+)', filename)
  if not match:
    raise ValueError(f"Cannot extract shape from filename: {filename}")

  height, width = map(int, match.groups())

  data = load_data(filename)
  num_patterns = data.shape[0]

  # Reshape each pattern to 2D
  reshaped_data = [pattern.reshape(height, width) for pattern in data]

  print(f"\nLoaded {filename}:")
  print(f"- Number of patterns: {num_patterns}")
  print(f"- Pattern shape: {height}x{width}")

  return np.array(reshaped_data), (height, width)


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


def process_raw_to_custom_image(raw_dir='raw_images_to_process',
    output_dir='custom_input_images'):
  """Process raw images to -1/1 CSV format - one flattened row"""
  os.makedirs(output_dir, exist_ok=True)

  for filename in os.listdir(raw_dir):
    if filename.endswith(('.png', '.jpg', '.jpeg')):
      # Load and preprocess image
      img = io.imread(os.path.join(raw_dir, filename))

      # Handle different image formats
      if len(img.shape) == 3:
        if img.shape[2] == 4:   # RGBA
          img = img[..., :3]    # Remove alpha channel
        img = rgb2gray(img)

      height, width = img.shape

      # Process to -1/1 and flatten
      img_binary = (img > np.mean(img)).astype(int)
      img_final = 2 * img_binary - 1
      img_flat = img_final.flatten()  # Flatten to one row

      # Save as CSV
      base_name = os.path.splitext(filename)[0]
      output_name = f"{base_name}_{height}x{width}.csv"
      np.savetxt(os.path.join(output_dir, output_name),
                 img_flat.reshape(1, -1),  # Ensure single row
                 delimiter=',',
                 fmt='%d')

      print(f"Processed {filename} -> {output_name}")
