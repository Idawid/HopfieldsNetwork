# -*- coding: utf-8 -*-
"""
Adapted code to read CSV files from 'input_images' directory and process them with Hopfield Network.
"""

import os
import numpy as np
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


def map_to_binary(data_flat):
    return [(d + 1) // 2 for d in data_flat]


def plot(data_flat, test_flat, predicted_flat, shape, figsize=(15, 5)):
    # Map -1 to 0 and 1 to 1 for binary visualization
    data_binary = map_to_binary(data_flat)
    test_binary = map_to_binary(test_flat)
    predicted_binary = map_to_binary(predicted_flat)

    data = [np.reshape(d, shape) for d in data_binary]
    test = [np.reshape(d, shape) for d in test_binary]
    predicted = [np.reshape(d, shape) for d in predicted_binary]

    num_images = len(data)
    fig, axarr = plt.subplots(num_images, 3, figsize=figsize)

    if num_images == 1:
        axarr = np.expand_dims(axarr, axis=0)

    for i in range(num_images):
        if i == 0:
            axarr[i, 0].set_title('Train Data')
            axarr[i, 1].set_title("Corrupted Input")
            axarr[i, 2].set_title('Predicted Output')

        axarr[i, 0].imshow(data[i], cmap='gray', vmin=0, vmax=1)
        axarr[i, 0].axis('off')
        axarr[i, 1].imshow(test[i], cmap='gray', vmin=0, vmax=1)
        axarr[i, 1].axis('off')
        axarr[i, 2].imshow(predicted[i], cmap='gray', vmin=0, vmax=1)
        axarr[i, 2].axis('off')

    plt.tight_layout()
    plt.savefig("output/result.png")
    plt.show()

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


def main():
    # Load data from CSV files in 'input_images' directory
    data = []
    input_dir = 'input_images'
    for filename in os.listdir(input_dir):
        if filename.endswith('.csv'):
            filepath = os.path.join(input_dir, filename)
            # Read CSV file
            d = np.loadtxt(filepath, delimiter=',')
            data.append(d)

    # Find maximum dimensions to pad images
    max_rows = max([d.shape[0] for d in data])
    max_cols = max([d.shape[1] for d in data])

    # Pad images to have the same dimensions
    padded_data = []
    for d in data:
        rows, cols = d.shape
        pad_top = (max_rows - rows) // 2
        pad_bottom = max_rows - rows - pad_top
        pad_left = (max_cols - cols) // 2
        pad_right = max_cols - cols - pad_left
        d_padded = np.pad(d, ((pad_top, pad_bottom), (pad_left, pad_right)), 'constant', constant_values=-1)
        padded_data.append(d_padded)

    # Flatten the images for processing
    data_flat = [np.reshape(d, (-1)) for d in padded_data]

    # Create Hopfield Network Model
    model = HopfieldNetwork()
    model.train_hebb(data_flat)

    # Generate test set by corrupting the input data
    test_flat = [get_corrupted_input(d, 0.3) for d in data_flat]

    # Predict the output using the Hopfield Network
    predicted_flat = model.predict(test_flat, threshold=0, asyn=False)
    print("Show prediction results...")

    # Calculate accuracy
    per_image_accuracy, overall_accuracy = calculate_accuracy(data_flat,
                                                              predicted_flat)
    for idx, acc in enumerate(per_image_accuracy):
        print(f"Image {idx + 1} Accuracy: {acc:.2f}%")
    print(f"Overall Accuracy: {overall_accuracy:.2f}%")

    # Plot the results
    shape = (max_rows, max_cols)
    plot(data_flat, test_flat, predicted_flat, shape)

    # Visualize the weights
    print("Show network weights matrix...")
    # model.plot_weights()

if __name__ == '__main__':
    main()
