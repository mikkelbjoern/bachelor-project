#!/usr/bin/env python3

# This script takes care of building the dynamic content of the report.
# If a specific part is specified, it will only build that part.
# If no part is specified, it will build all parts.

import click
import os
from src.symlink_models import symlink_models
from src.build_kernel_example import build_kernel_examples
from src.build_convolution_example import build_convolution_example
from src.build_confounder_label_correlation import build_confounder_label_correlation
from src.build_saliency_maps import build_saliency_maps
from src.build_prediction_strength import build_prediction_strength

parts = [
    "convolution_example",
    "kernel_examples",
    "confounder_label_correlation",
    "saliency_maps",
    "prediction_strength",
]

BUILD_FOLDER = os.path.dirname(os.path.realpath(__file__)) + "/build"
IMAGE_FOLDER = os.path.dirname(os.path.realpath(__file__)) + "/images"

# Use the click library to define the command line interface
@click.command()
@click.option("--part", default=None, help="Part to build")
def main(part):
    # Make sure the build folder exists
    if not os.path.exists(BUILD_FOLDER):
        print("No build folder found. Creating...")
        os.makedirs(BUILD_FOLDER)

    symlink_models()

    # Build the report
    if part is None:
        for part in parts:
            build_part(part)
    else:
        build_part(part)


def build_part(part):
    # Make sure that there exists a part folder
    part_folder = BUILD_FOLDER + "/" + part
    if not os.path.exists(part_folder):
        os.makedirs(part_folder)
    else:
        # Remove the old files
        for file in os.listdir(part_folder):
            path = part_folder + "/" + file
            if os.path.isfile(path):
                os.remove(path)
            
            # Check if the path is a symlink
            if os.path.islink(path):
                os.remove(path)

    # Change to the part folder
    os.chdir(part_folder)

    # Build the part
    if part == "convolution_example":
        build_convolution_example()

    if part == "kernel_examples":
        build_kernel_examples()

    if part == "confounder_label_correlation":
        build_confounder_label_correlation()

    if part == "saliency_maps":
        build_saliency_maps()
    
    if part == "prediction_strength":
        build_prediction_strength()




if __name__ == "__main__":
    main()
