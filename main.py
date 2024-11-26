# -*- coding: utf-8 -*-
"""
Adapted code to read CSV files from 'input_images' directory and process them with Hopfield Network.
"""

from data_loader import get_csv_files, get_custom_input, generate_custom_input, \
    process_raw_to_custom_image
from config import get_config
from hopfield_runner import run_hopfield

if __name__ == '__main__':
    # Uncomment to regenerate random inputs
    generate_custom_input(shapes=[                  # neuron/shape Ratio
        (32, 32, 4),    # 6 patterns of 32x32         0.0039
        (16, 16, 8),    # 4 patterns of 16x16         0.0313
        (8, 8, 4),      # 4 patterns of 16x16         0.0625
        (8, 8, 8),      # 8 patterns of 16x16         0.125
        (8, 8, 16),     # 16 patterns of 8x8 will test limits of memory 0.25
    ])
    process_raw_to_custom_image()

    CONFIG = get_config()
    files_to_process = get_csv_files() # move files between dontprocess/input images

    files_to_process += get_custom_input() # comment out to not process custom

    for filename in files_to_process:
        run_hopfield(filename, CONFIG)