"""

This script creates a .csv file containing each xray image and its corresponding label in the training/validation set.
Re-run this script every time you add more images.

Author: Luke Farritor 10/25/20

"""

from pathlib import Path
import csv

folders = ["body_not_body_segmented", "augmented"]
unlabeled_ending = ".png"
labeled_ending = "-labeled.png"
csv_name = "images.csv"

images_dict = {}


def find_images():
    for folder in folders:
        # add unlabeled image paths to dictionary
        for path in Path(folder).rglob("*" + unlabeled_ending):
            if not str(path).endswith(labeled_ending):
                images_dict[str(path)] = None

        # find and add labeled image paths to corresponding unlabeled image in dict
        for path in Path(folder).rglob("*" + labeled_ending):
            corresponding_unlabeled = str(path)[:-len(labeled_ending)] + unlabeled_ending
            if corresponding_unlabeled in images_dict:
                images_dict[corresponding_unlabeled] = str(path)

        # write dictionary of image paths to csv file
        try:
            with open(csv_name, 'w') as csv_file:
                w = csv.writer(csv_file)
                w.writerows(images_dict.items())
        except IOError:
            print("I/O error")

    print("Done, " + str(len(images_dict)) + " image pairs were written to " + csv_name)


if __name__ == "__main__":
    find_images()
