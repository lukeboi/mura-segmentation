import random

import pandas as pd
from PIL import Image as pImage
import PIL
from tensorflow.keras.preprocessing.image import load_img
import numpy as np

class ImageLoader():
    def __init__(self, image_size, image_paths, perform_augmentations=True):
        self.perform_augmentations = perform_augmentations
        self.image_size = image_size
        self.images = image_paths

    def get_random_batch(self, batch_size):
        #randomly select images
        batch_input_img_paths = []
        batch_target_img_paths = []
        seeds = []
        for i in range(0, batch_size):
            i = random.choice(list(self.images.items()))
            batch_input_img_paths.append(i[0])
            batch_target_img_paths.append(i[1])
            seeds.append(random.randint(0, 10000))


        x = np.zeros((batch_size,) + self.image_size + (1,), dtype="float32")
        for j, path in enumerate(batch_input_img_paths):
            img = load_img(path, color_mode="grayscale")
            img = self.perform_augmentation(img, seeds[j])

            # pad the image to 512
            img_arr = np.array(img)
            img_arr = np.pad(img_arr,
                             ((0, self.image_size[0] - img_arr.shape[0]), (0, self.image_size[1] - img_arr.shape[1])),
                             mode='constant')

            # set the image in the array
            x[j] = np.expand_dims(img_arr / 255, 2)


        y = np.zeros((batch_size,) + self.image_size + (1,), dtype="uint8")
        for j, path in enumerate(batch_target_img_paths):
            img = load_img(path, color_mode="grayscale")
            img = self.perform_augmentation(img, seeds[j])

            # pad the image to 512
            img_arr = np.array(img)
            img_arr = np.pad(img_arr,
                             ((0, self.image_size[0] - img_arr.shape[0]), (0, self.image_size[1] - img_arr.shape[1])),
                             mode='constant')

            # set the image in the array
            y[j] = np.expand_dims(img_arr / 255, 2)

        return x, y

    def perform_augmentation(self, img, seed):
        if not self.perform_augmentations:
            return img

        random.seed(seed)

        if int(random.randint(0, 1)):
            # horizontal flip
            img = img.transpose(PIL.Image.FLIP_LEFT_RIGHT)

        if int(random.randint(0,1)):
            # vertical fip
            img = img.transpose(PIL.Image.FLIP_TOP_BOTTOM)

        # random rotate
        img.rotate(random.randint(0, 359))

        if random.randint(0, 5) == 0:
            # horizontal shrink
            img = img.resize((int(img.size[0] / 2), img.size[1]))
        if random.randint(0, 5) == 0:
            # vertical shrink
            img = img.resize((img.size[0], int(img.size[1] / 2)))

        # display image for debugging purposes
        # img.show()
        return img

    def pad_image(self, image_size):
        return None