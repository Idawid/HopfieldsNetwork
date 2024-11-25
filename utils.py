# Utils
import numpy as np


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
