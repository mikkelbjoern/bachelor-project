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
from src.build_saliency_maps import build_saliency_maps, build_only_lesion_saliency_maps
from src.build_prediction_strength import build_prediction_strength
from src.build_near_neigh import build_near_neigh
from src.build_segmented_prediction_strength import build_segmented_prediction_strength
from src.build_segmented_images_example import build_segmented_images_examlpe
from src.build_data_aug_examples import build_data_aug_examples
import difflib

# The second argument in the tuple is weather the part is default built
parts_and_default = [
    ("convolution_example", True),
    ("kernel_examples", True),
    ("confounder_label_correlation", True),
    ("saliency_maps", False),
    ("only_lesion_saliency_maps", False),
    ("prediction_strength", True),
    ("near_neigh", False),
    ("segmented_prediction_strength", True),
    ("segmented_images_example", True),
    ("data_aug_examples", True)
]
parts = [part for part, _ in parts_and_default]

BUILD_FOLDER = os.path.dirname(os.path.realpath(__file__)) + "/build"
IMAGE_FOLDER = os.path.dirname(os.path.realpath(__file__)) + "/images"

# Use the click library to define the command line interface
@click.command()
@click.option("--part", default=None, help="Part to build")
def main(part):
    if not part in parts and part != 'all':
        print(f"Invalid part: '{part}'")
        close_matches = difflib.get_close_matches(part, parts.append('all'))
        if len(close_matches) > 0:
            print("Did you mean one of these?")
            for close_match in close_matches:
                print(f"\t{close_match}")
        exit(1)

    if part != 'all':
        print(f"Building part: '{part}'")
    else:
        print("Building all parts")

    # Make sure the build folder exists
    if not os.path.exists(BUILD_FOLDER):
        print("No build folder found. Creating...")
        os.makedirs(BUILD_FOLDER)

    symlink_models()

    # Build the report
    if part is None:
        for part, default_build in parts_and_default:
            if default_build:
                build_part(part)
    elif part == 'all':
        for part, _ in parts_and_default:
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
    
    if part == "near_neigh":
        build_near_neigh()

    if part == "segmented_prediction_strength":
        build_segmented_prediction_strength()
    
    if part == "segmented_images_example":
        build_segmented_images_examlpe()
    
    if part == "only_lesion_saliency_maps":
        build_only_lesion_saliency_maps()

    if part == "data_aug_examples":
        build_data_aug_examples()


if __name__ == "__main__":
    main()
