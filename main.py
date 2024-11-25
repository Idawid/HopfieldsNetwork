# -*- coding: utf-8 -*-
"""
Adapted code to read CSV files from 'input_images' directory and process them with Hopfield Network.
"""
import os

import numpy as np

from data_loader import get_csv_files, load_csv_patterns, ensure_experiment_dir
from experiments import get_experiments
from visualizer import plot, save_plot

np.random.seed(1)
from matplotlib import pyplot as plt
from hopfield_network import HopfieldNetwork

# Utils
def get_corrupted_input(input_data, corruption_level):
    corrupted = np.copy(input_data)
    inv = np.random.binomial(n=1, p=corruption_level, size=len(input_data))
    for i, v in enumerate(input_data):
        if inv[i]:
            corrupted[i] = -1 * v
    return corrupted


def calculate_accuracy(original, predicted):
    per_image_accuracy = []
    total_correct = 0
    total_pixels = 0

    for orig, pred in zip(original, predicted):
        correct = np.sum(orig == pred)
        total_correct += correct
        total_pixels += len(orig)
        accuracy = correct / len(orig) * 100
        per_image_accuracy.append(accuracy)

    overall_accuracy = (total_correct / total_pixels) * 100
    return per_image_accuracy, overall_accuracy


def generate_random_patterns(n_patterns, shape=(5, 5)):
    """Generate n random binary patterns of given shape"""
    patterns = []
    for _ in range(n_patterns):
        pattern = np.random.choice([-1, 1], size=(shape[0] * shape[1]))
        patterns.append(pattern)
    return patterns


def run_hopfield(filename, experiment_config):
    """Run Hopfield Network experiment for a single CSV file"""
    print(f"\n=== Processing {filename} for {experiment_config['name']} ===")

    # Create experiment directories
    dirs = ensure_experiment_dir(experiment_config['name'])

    patterns, pattern_shape = load_csv_patterns(filename)

    # Process for each learning rule separately
    if experiment_config['use_hebb']:
        print("\nRunning Hebb's rule...")
        model = HopfieldNetwork()
        model.train_hebb(patterns)
        run_tests_and_save(
            model, patterns, pattern_shape, filename,
            experiment_config, dirs['hebb'], 'hebb'
        )

    if experiment_config['use_oja']:
        print("\nRunning Oja's rule...")
        model = HopfieldNetwork()
        model.train_oja(
            patterns,
            learning_rate=experiment_config['learning_rate_oja'],
            epochs=experiment_config['epochs_oja']
        )
        run_tests_and_save(
            model, patterns, pattern_shape, filename,
            experiment_config, dirs['oja'], 'oja'
        )


def run_tests_and_save(model, patterns, pattern_shape, filename,
    experiment_config, dirs, rule_name):
    """Run tests and save results for a specific learning rule"""
    results = {}
    corruption_levels = experiment_config.get('corruption_levels',
                                              [experiment_config[
                                                   'corruption_level']])

    for corruption_level in corruption_levels:
        print(f"\nTesting with corruption level: {corruption_level}")
        test_patterns = [get_corrupted_input(p, corruption_level)
                         for p in patterns]

        predicted_patterns = model.predict(
            test_patterns,
            threshold=experiment_config['threshold'],
            asyn=experiment_config['async_update']
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
    if experiment_config['save_plots'] or experiment_config['show_plots']:
        for corruption_level, result in results.items():
            fig = plot(
                patterns,
                result['test_patterns'],
                result['predicted_patterns'],
                pattern_shape,
                figsize=(20, 3 * len(patterns))
            )

            if experiment_config['save_plots']:
                save_plot(fig,
                          f'{filename[:-4]}_corr{corruption_level}_patterns.png',
                          output_dir=dirs['plots'])

            if not experiment_config['show_plots']:
                plt.close(fig)

    # Weights visualization
    if experiment_config['save_plots']:
        plt.figure(figsize=(6, 5))
        w_mat = plt.imshow(model.W, cmap=plt.cm.coolwarm)
        plt.colorbar(w_mat)
        plt.title(f"Network Weights - {rule_name}")
        plt.tight_layout()
        save_plot(plt.gcf(), f'{filename[:-4]}_weights.png',
                  output_dir=dirs['weights'])
        plt.close()


if __name__ == '__main__':
    # Get all experiments or select specific ones
    experiments = get_experiments()

    # Run all experiments
    files_to_process = get_csv_files()

    for exp_name, exp in experiments.items():
        print(f"\n=== Starting Experiment: {exp_name} ===")
        for filename in files_to_process:
            run_hopfield(filename, exp.config)
