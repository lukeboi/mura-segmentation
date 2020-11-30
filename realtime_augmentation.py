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
        # randomly select images
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
            img = pImage.open(path)
            img = self.perform_augmentation(img, seeds[j], self.image_size)

            # pad the image to 512
            img_arr = np.array(img)
            img_arr = np.pad(img_arr,
                             ((0, self.image_size[0] - img_arr.shape[0]), (0, self.image_size[1] - img_arr.shape[1])),
                             mode='constant')

            # set the image in the array
            x[j] = np.expand_dims(img_arr / 255, 2)

        y = np.zeros((batch_size,) + self.image_size + (1,), dtype="uint8")
        for j, path in enumerate(batch_target_img_paths):
            img = pImage.open(path)

            # apply a threshold to the mask images so they're only black or white
            threshold = 128
            img = img.point(lambda p: p > threshold and 255)

            img = self.perform_augmentation(img, seeds[j], self.image_size)

            # pad the image to 512
            img_arr = np.array(img)
            img_arr = np.pad(img_arr,
                             ((0, self.image_size[0] - img_arr.shape[0]), (0, self.image_size[1] - img_arr.shape[1])),
                             mode='constant')

            # set the image in the array
            y[j] = np.expand_dims(img_arr / 255, 2)

        # display images for debugging purposes
        # pImage.fromarray(np.uint8(x[0][:, :, 0] * 255), mode="L").show()
        # pImage.fromarray(np.uint8(y[0][:, :, 0] * 255), mode="L").show()

        return x, y

    def perform_augmentation(self, img, seed, image_size):
        # pad image to 512
        padded_bg = pImage.new("L", image_size, 0)
        padded_bg.paste(img)
        img = padded_bg

        if not self.perform_augmentations:
            return img

        random.seed(seed)

        if int(random.randint(0, 1)):
            # horizontal flip
            img = img.transpose(PIL.Image.FLIP_LEFT_RIGHT)

        if int(random.randint(0, 1)):
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
