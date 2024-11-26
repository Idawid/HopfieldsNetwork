# hopfield_runner.py
import os
import numpy as np
import matplotlib.pyplot as plt
from hopfield_network import HopfieldNetwork
from utils import get_corrupted_input, calculate_accuracy
from data_loader import load_csv_patterns
from visualizer import plot, save_plot


def run_hopfield(filename, config):
  """Run Hopfield Network experiment for a single CSV file"""
  print(f"\n=== Processing {filename} ===")

  # Create output directories
  output_dirs = {
    'hebb': {
      'plots': os.path.join('output', 'hebb', 'plots'),
      'weights': os.path.join('output', 'hebb', 'weights'),
      'results': os.path.join('output', 'hebb', 'results')
    },
    'oja': {
      'plots': os.path.join('output', 'oja', 'plots'),
      'weights': os.path.join('output', 'oja', 'weights'),
      'results': os.path.join('output', 'oja', 'results')
    }
  }

  # Create directories if they don't exist
  for rule_dirs in output_dirs.values():
    for directory in rule_dirs.values():
      if not os.path.exists(directory):
        os.makedirs(directory)

  patterns, pattern_shape = load_csv_patterns(filename)

  # Process for each learning rule
  if config['use_hebb']:
    print("\nRunning Hebb's rule...")
    model = HopfieldNetwork()
    model.train_hebb(patterns)
    run_tests_and_save(
        model, patterns, pattern_shape, filename,
        config, output_dirs['hebb'], 'hebb'
    )

  if config['use_oja']:
    print("\nRunning Oja's rule...")
    model = HopfieldNetwork()
    model.train_oja(
        patterns,
        learning_rate=config['learning_rate_oja'],
        epochs=config['epochs_oja']
    )
    run_tests_and_save(
        model, patterns, pattern_shape, filename,
        config, output_dirs['oja'], 'oja'
    )


def run_tests_and_save(model, patterns, pattern_shape, filename, config, dirs,
    rule_name):
  """Run tests and save results for a specific learning rule"""
  results = {}

  # Test pattern stability (no corruption)
  stable_patterns = []
  for i, pattern in enumerate(patterns):
    output = model.predict([pattern], threshold=config['threshold'],
                           asyn=config['async_update'])[0]
    is_stable = np.array_equal(pattern, output)
    stable_patterns.append(is_stable)

  # Calculate pattern similarities
  similarities = []
  for i in range(len(patterns)):
    for j in range(i + 1, len(patterns)):
      sim = np.sum(patterns[i] == patterns[j]) / len(patterns[i])
      similarities.append(sim)

  # Save comprehensive results
  with open(
      os.path.join(dirs['results'], f'{filename[:-4]}_{rule_name}_results.txt'),
      'w') as f:
    # Basic information
    f.write(f"=== Analysis Results ===\n\n")
    f.write(f"File: {filename}\n")
    f.write(f"Learning Rule: {rule_name}\n")
    f.write(f"Pattern Set Info:\n")
    f.write(f"- Number of patterns: {len(patterns)}\n")
    f.write(f"- Pattern shape: {pattern_shape}\n\n")

    # Stability analysis
    f.write(f"Stability Analysis:\n")
    f.write(f"- Stable patterns: {sum(stable_patterns)}/{len(patterns)}\n")
    f.write(f"- Stability ratio: {sum(stable_patterns) / len(patterns):.2f}\n")
    f.write("- Per-pattern stability:\n")
    for i, is_stable in enumerate(stable_patterns):
      f.write(f"  Pattern {i + 1}: {'Stable' if is_stable else 'Unstable'}\n")
    f.write("\n")

    # Pattern similarities
    f.write(f"Pattern Similarities:\n")
    f.write(f"- Mean: {np.mean(similarities):.3f}\n")
    f.write(f"- Std: {np.std(similarities):.3f}\n")
    f.write(f"- Min: {np.min(similarities):.3f}\n")
    f.write(f"- Max: {np.max(similarities):.3f}\n\n")

    # Network statistics
    f.write(f"Network Statistics:\n")
    f.write(f"Weight matrix:\n")
    f.write(f"- Mean: {np.mean(model.W):.3f}\n")
    f.write(f"- Std: {np.std(model.W):.3f}\n")
    f.write(f"- Min: {np.min(model.W):.3f}\n")
    f.write(f"- Max: {np.max(model.W):.3f}\n\n")

    # Corruption tests
    f.write(f"Corruption Tests:\n")
    for corruption_level in config['corruption_levels']:
      f.write(f"\nCorruption Level: {corruption_level}\n")

      # Create corrupted test patterns
      test_patterns = [get_corrupted_input(p, corruption_level)
                       for p in patterns]

      # Run prediction
      predicted_patterns = model.predict(
          test_patterns,
          threshold=config['threshold'],
          asyn=config['async_update']
      )

      # Calculate accuracy
      per_pattern_accuracy, overall_accuracy = calculate_accuracy(
          patterns, predicted_patterns)

      # Save results
      f.write(f"- Overall Accuracy: {overall_accuracy:.2f}%\n")
      f.write("- Per-pattern Accuracy:\n")
      for idx, acc in enumerate(per_pattern_accuracy):
        f.write(f"  Pattern {idx + 1}: {acc:.2f}%\n")

      # Store results for visualization
      results[corruption_level] = {
        'per_pattern_accuracy': per_pattern_accuracy,
        'overall_accuracy': overall_accuracy,
        'test_patterns': test_patterns,
        'predicted_patterns': predicted_patterns
      }

  # Visualization code remains the same...
  if config['save_plots'] or config['show_plots']:
    for corruption_level, result in results.items():
      fig = plot(
          patterns,
          result['test_patterns'],
          result['predicted_patterns'],
          pattern_shape,
          figsize=(20, 3 * len(patterns))
      )

      if config['save_plots']:
        save_plot(fig,
                  f'{filename[:-4]}_corr{corruption_level}_patterns.png',
                  output_dir=dirs['plots'])

      if not config['show_plots']:
        plt.close(fig)

  # Weights visualization
  if config['save_plots']:
    plt.figure(figsize=(6, 5))
    w_mat = plt.imshow(model.W, cmap=plt.cm.coolwarm)
    plt.colorbar(w_mat)
    plt.title(f"Network Weights - {rule_name}")
    plt.tight_layout()
    save_plot(plt.gcf(), f'{filename[:-4]}_weights.png',
              output_dir=dirs['weights'])
    plt.close()

