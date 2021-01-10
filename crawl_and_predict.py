from pathlib import Path

from keras_preprocessing.image import load_img

import prediction_feedback_gui
import realtime_augmentation
from tensorflow.python.keras.models import Model
from keras.models import load_model
from tensorflow import keras
from PIL import Image as pImage
from PIL import ImageEnhance
import numpy as np
import tensorflow as tf
import os

# setup GPU
config = tf.compat.v1.ConfigProto(gpu_options=
                                  tf.compat.v1.GPUOptions(per_process_gpu_memory_fraction=0.6)
                                  # device_count = {'GPU': 1}
                                  )
config.gpu_options.allow_growth = True
session = tf.compat.v1.Session(config=config)
tf.compat.v1.keras.backend.set_session(session)
gpus = tf.config.experimental.list_physical_devices('GPU')


dir_to_crawl = "images_to_predict"
dir_to_save = "successful_predictions/"
labeled_image_name = "-labeled.png"

all_images = []

image_size = (512, 512)
batch_size = 1

class ImageHandler(keras.utils.Sequence):
    def __init__(self, batch_size, image_size, image_paths, size, do_augs):
        self.imageLoader = realtime_augmentation.ImageLoader(image_size, image_paths, do_augs)
        self.size = size
        self.image_size = image_size

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        img = load_img(path, color_mode="grayscale")

        # pad the image to 512
        img_arr = np.array(img)
        img_arr = np.pad(img_arr,
                         ((0, self.image_size[0] - img_arr.shape[0]), (0, self.image_size[1] - img_arr.shape[1])),
                         mode='constant')

        # set the image in the array
        return np.expand_dims(img_arr / 255, 2)


for path in Path(dir_to_crawl).rglob("*.png"):
    if str(path).endswith(".png"):
        all_images.append(str(path))

model = load_model("trained_model")

# brightness = ImageEnhance.Brightness()

for i in all_images:
    img = load_img(i, color_mode="grayscale")

    # pad the image to 512
    img_arr = np.array(img)
    img_arr = np.pad(img_arr,
                     ((0, image_size[0] - img_arr.shape[0]), (0, image_size[1] - img_arr.shape[1])),
                     mode='constant')

    # set the image in the array
    prediction = model.predict(np.expand_dims(img_arr / 255, 2).reshape(1, 512, 512, 1))

    # normalize colors to all black and all white
    predicted_image = pImage.fromarray(np.uint8(prediction[0, :, :, 0] * 255), mode="L")
    predicted_image = np.array(predicted_image)
    predicted_image = pImage.fromarray(np.uint8((predicted_image) * 1), mode="L")


    # load original image for displaying
    original_img = pImage.new("L", image_size)
    original_img.paste(pImage.Image.convert(pImage.open(i), mode="L"), (0, 0))

    # let user edit image
    prediction_window = prediction_feedback_gui.UserImageFeedbackWindow(predicted_image, original_img)

    predicted_image = prediction_window.get_edited_image()

    if prediction_window.should_move_image():
        # save image and move to training set folder
        Path(os.path.dirname(dir_to_save + i)).mkdir(parents=True, exist_ok=True)
        predicted_image.save(dir_to_save + i[:-4] + labeled_image_name)
        img.save(dir_to_save + i)
        os.remove(i)
