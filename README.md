# Mura X-Ray Segmentation Project

This project is an implementation of the Unet Image Segmenetation CNN for use on the Stanford Mura Dataset. The model is trained to segment between the body and not body parts of an X-Ray image. The project uses Keras and Tensorflow for ML and uses PyQt for a training feedback GUI. I can't include the full dataset and segmentations in this github respository (due to both storage and licensing constraints), but here's a few examples of what my trained model does:

Input image on left, trained network output on right

![example_image](https://lukefarritor.com/img/mura_example.png)

All the code in this reposotory was written by yours truly. I worked on this project independently for Eric Psota as an undergraduate researcher at UNL. Dr. Psota provided general direction and advice on a weekly basis and also helped me on the occasion I was stuck or confused by Tensorflow's Documentation. We originally planned to use the segmented dataset as part of a larger project relating to Radiology. However, for reasons unrelated to this project, Dr. Psota left UNL in Jan 2021. There's a good chance I'll continue the project on my own, but as of this writing I'm not sure. For the time being, I'm making this repo public for two reasons:

1) To provide more example code to those learning about Machine Learning
2) To demonstrate my ML chops to recruiters

Here's an overview of what each script does:

main.py - the primary training script. Defines the model, loads the dataset (as defined in images.csv by get_images_and_labels.py), trains the network (takes overnight on my 1650ti), displays validation results to the user, and saves the trained network output to disk.

get_images_and_labels.py - loads the dataset, stored in /body_not_body_segmented/, shuffles the order, and saves the resulting image/mask map to images.csv

crawl_and_predict.py - loads the trained network, creates predictions for each image in /images_to_predict/, displays those predictions to the using using prediction_feedback_gui.py, and if the users chooses will save the accurate image/mask combo to /successful_predictions/. The user can then take these successful predictions and move them to the main dataset, /body_not_body_segmented/.

prediction_feedback_gui.py - helper functions for a GUI that displays a predicted mask for a given image. The user can then draw on the mask to correct any inaccuracies and choose to save the resultant mask to be put in the dataset.

realtime_augmentation.py - image loader. loads images as defined in images.csv, augments them randomly (scale, offset, flipping, rotation, brightness, etc) and defines a class to handle images as defined by main.py

augmentation_generation.py - unused. it was a prior (much worse) attempt at augmentation.

All rights reserved.
