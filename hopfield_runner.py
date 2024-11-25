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

  for corruption_level in config['corruption_levels']:
    print(f"\nTesting with corruption level: {corruption_level}")
    test_patterns = [get_corrupted_input(p, corruption_level)
                     for p in patterns]

    predicted_patterns = model.predict(
        test_patterns,
        threshold=config['threshold'],
        asyn=config['async_update']
    )

    per_pattern_accuracy, overall_accuracy = calculate_accuracy(
        patterns, predicted_patterns)

    results[corruption_level] = {
      'per_pattern_accuracy': per_pattern_accuracy,
      'overall_accuracy': overall_accuracy,
      'test_patterns': test_patterns,
      'predicted_patterns': predicted_patterns
    }

    print(
        f"\nAccuracy Results ({rule_name}, corruption level {corruption_level}):")
    print("-" * 20)
    for idx, acc in enumerate(per_pattern_accuracy):
      print(f"Pattern {idx + 1} Accuracy: {acc:.2f}%")
    print(f"Overall Accuracy: {overall_accuracy:.2f}%")

    # Save accuracy results
    with open(os.path.join(dirs['results'],
                           f'{filename[:-4]}_corr{corruption_level}_accuracy.txt'),
              'w') as f:
      f.write(f"File: {filename}\n")
      f.write(f"Learning Rule: {rule_name}\n")
      f.write(f"Corruption Level: {corruption_level}\n")
      f.write("-" * 20 + "\n")
      for idx, acc in enumerate(per_pattern_accuracy):
        f.write(f"Pattern {idx + 1} Accuracy: {acc:.2f}%\n")
      f.write(f"Overall Accuracy: {overall_accuracy:.2f}%\n")

  # Visualization
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
