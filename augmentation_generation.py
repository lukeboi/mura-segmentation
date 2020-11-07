"""

This script creates a .csv file containing each xray image and its corresponding label in the training/validation set.
Re-run this script every time you add more images.

Author: Luke Farritor 11/4/20

"""
import copy
import os
from pathlib import Path
import pandas as pd
import PIL
from PIL import Image

csv_name = "images.csv"
source_folder_name = "body_not_body_segmented"
augmented_folder_name = "augmented"
unlabeled_ending = ".png"
labeled_ending = "-labeled.png"
augmentation_names = ["h_flip", "v_flip", "hv_flip", "90_rot", "180_rot", "270_rot", "h_shrink", "v_shrink"]

# todo: delete augmentation folders, run csv findings again, then generate augmentations again

def get_unaugmented_images():
    images = []
    # add unlabeled image paths to dictionary
    for path in Path(source_folder_name).rglob("*" + unlabeled_ending):
        if not str(path).endswith(labeled_ending):
            images.append(str(path))

    # find and add labeled image paths to corresponding unlabeled image in dict
    for path in Path(source_folder_name).rglob("*" + labeled_ending):
        corresponding_unlabeled = str(path)[:-len(labeled_ending)] + unlabeled_ending
        if corresponding_unlabeled in images:
            images.append(str(path))

    return images


source_images = get_unaugmented_images()

for i in source_images:
    img = Image.open(i)
    for aug in augmentation_names:
        # load image directory
        dir = ".\\" + augmented_folder_name + "\\" + aug + "\\" + i
        os.makedirs(os.path.split(dir)[0], exist_ok=True)

        if aug == "h_flip":
            img_aug = img.transpose(PIL.Image.FLIP_LEFT_RIGHT)
        elif aug == "v_flip":
            img_aug = img.transpose(PIL.Image.FLIP_TOP_BOTTOM)
        elif aug == "hv_flip":
            img_aug = img.transpose(PIL.Image.FLIP_TOP_BOTTOM)
            img_aug = img.transpose(PIL.Image.FLIP_LEFT_RIGHT)
        elif aug == "90_rot":
            img_aug = img.transpose(PIL.Image.ROTATE_90)
        elif aug == "180_rot":
            img_aug = img.transpose(PIL.Image.ROTATE_180)
        elif aug == "270_rot":
            img_aug = img.transpose(PIL.Image.ROTATE_270)
        elif aug == "h_shrink":
            img_aug = img.resize((int(img.size[0] / 2), img.size[1]))
        elif aug == "v_shrink":
            img_aug = img.resize((img.size[0], int(img.size[1] / 2)))

        img_aug.save(dir)

