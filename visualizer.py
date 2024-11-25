import numpy as np
import matplotlib.pyplot as plt
import os


def map_to_binary(data_flat):
  """Convert -1/1 to 0/1 for visualization"""
  return [(d + 1) // 2 for d in data_flat]


def plot(data_flat, test_flat, predicted_flat, shape, figsize=None):
  """
  Plot all patterns from a single file in one figure.
  Each row represents one pattern with its corrupted and predicted versions.
  """
  # Map -1 to 0 and 1 to 1 for binary visualization
  data_binary = map_to_binary(data_flat)
  test_binary = map_to_binary(test_flat)
  predicted_binary = map_to_binary(predicted_flat)

  num_patterns = len(data_flat)

  # Calculate appropriate figure size if not provided
  if figsize is None:
    height_per_pattern = 2  # inches per pattern row
    figsize = (20, height_per_pattern * num_patterns)

  # Create figure with subplots for each pattern
  fig, axarr = plt.subplots(num_patterns, 5, figsize=figsize)

  # Handle single pattern case
  if num_patterns == 1:
    axarr = np.expand_dims(axarr, axis=0)

  # Titles for columns
  plt.rcParams.update({'font.size': 20})  # Increase base font size
  titles = ['Original', 'Corrupted', 'Predicted',
            'Corruption Diff', 'Recovery Diff']

  # Add column titles
  for j, title in enumerate(titles):
    axarr[0, j].set_title(title)

  # Process each pattern
  for i in range(num_patterns):
    # Reshape patterns
    data = np.reshape(data_binary[i], shape)
    test = np.reshape(test_binary[i], shape)
    predicted = np.reshape(predicted_binary[i], shape)

    # Calculate differences
    corruption_diff = np.abs(data - test)
    recovery_diff = np.abs(data - predicted)

    # Create colored difference maps
    corruption_rgb = np.zeros((*shape, 3))
    recovery_rgb = np.zeros((*shape, 3))

    # Red for differences
    corruption_rgb[corruption_diff == 1] = [1, 0, 0]
    recovery_rgb[recovery_diff == 1] = [1, 0, 0]
    # Green/Gray for matches
    corruption_rgb[corruption_diff == 0] = [0.7, 0.7, 0.7]  # Gray for unchanged
    recovery_rgb[recovery_diff == 0] = [0, 1, 0]  # Green for correct recovery

    # Plot all versions
    images = [data, test, predicted, corruption_rgb, recovery_rgb]

    for j, img in enumerate(images):
      if j < 3:  # Original, corrupted, and predicted in grayscale
        axarr[i, j].imshow(img, cmap='gray', vmin=0, vmax=1)
      else:  # Difference maps in color
        axarr[i, j].imshow(img)
      axarr[i, j].axis('off')

    # Add accuracy information
    correct_pixels = np.sum(recovery_diff == 0)
    total_pixels = np.prod(shape)
    accuracy = (correct_pixels / total_pixels) * 100
    axarr[i, 0].set_ylabel(f'Pattern {i + 1}\nAcc: {accuracy:.1f}%',
                           rotation=0, ha='right', va='center')

  plt.tight_layout()
  return fig


def save_plot(fig, filename, output_dir='output'):
  """Save plot to file"""
  if not os.path.exists(output_dir):
    os.makedirs(output_dir)
  fig.savefig(os.path.join(output_dir, filename))
