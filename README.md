# Mura X-Ray Segmentation Project

This project is an implementation of the Segnet Image Segmenetation CNN for use on the Stanford Mura Dataset. The model is trained to segment between the body and not body parts of an X-Ray image. The project uses Keras and Tensorflow for ML and uses PyQt for a training feedback GUI. I can't include the full dataset and segmentations in this github respository (due to both storage and licenscing constraints), but here's a few examples of what my trained model does:

Input Image | Trained Network Output
----------- | ------------------
foo | bar

All the code in this reposotory was written by yours truly. I worked on this project independently for Eric Psota as an undergraduate researcher at UNL. Dr. Psota provided general direction and advice on a weekly basis and also helped me on the occasion I was stuck or confused by Tensorflow's Documentation. We originally planned to use the segmented dataset as part of a larger project relating to Radiology. However, for reasons unrelated to this project, Dr. Psota left UNL in Jan 2021. There's a good chance I'll continue the project on my own, but as of this writing I'm not sure. For the time being, I'm making this repo public for two reasons:

1) To provide more example code to those learning about Machine Learning
2) To demonstrate my ML chops to recruiters

Please note, I have yet to organize this code. There's magic numbers and randomly commented out blocks of code everywhere. Someday when I'm less busy I'll go through and clean things up. In the meantime, here's an overview of what each script does:

main.py - the primary training script. Defines the model, loads the dataset (as defined in get_images_and_labels.py), trains the network (takes overnight on my 1650ti) and saves the output
