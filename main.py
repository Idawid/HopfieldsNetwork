# -*- coding: utf-8 -*-
"""
Adapted code to read CSV files from 'input_images' directory and process them with Hopfield Network.
"""

from data_loader import get_csv_files, get_custom_input
from config import get_config
from hopfield_runner import run_hopfield

if __name__ == '__main__':
    # Uncomment to regenerate random inputs
    # generate_custom_input(shapes=[
    #     (32, 32, 6),  # 6 patterns of 32x32
    #     (16, 16, 4),  # 4 patterns of 16x16
    #     (8, 8, 10)    # 10 patterns of 8x8
    # ])

    CONFIG = get_config()
    files_to_process = get_csv_files() # move files between dontprocess/input images

    files_to_process += get_custom_input() # comment out to not process custom

    for filename in files_to_process:
        run_hopfield(filename, CONFIG)