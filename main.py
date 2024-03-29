"""

Predict xray and void sections of an xray image.

Author: Luke Farritor 10/26/20

"""
import pandas as pd
from IPython.display import Image, display
from keras_preprocessing.image import ImageDataGenerator
from tensorflow.keras.preprocessing.image import load_img
import PIL
from PIL import Image as pImage
from PIL import ImageOps
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.python.keras.models import Model
from tensorflow import keras
import tensorflow as tf
from keras.models import load_model
from tensorflow.python.keras.optimizer_v2.adam import Adam
import numpy as np
import realtime_augmentation
import datetime

# setup GPU
config = tf.compat.v1.ConfigProto(gpu_options=
                                  tf.compat.v1.GPUOptions(per_process_gpu_memory_fraction=0.6)
                                  # device_count = {'GPU': 1}
                                  )
config.gpu_options.allow_growth = True
session = tf.compat.v1.Session(config=config)
tf.compat.v1.keras.backend.set_session(session)
gpus = tf.config.experimental.list_physical_devices('GPU')

# gpus = tf.config.experimental.list_physical_devices('GPU')
# if gpus:
#   try:
#     # Currently, memory growth needs to be the same across GPUs
#     for gpu in gpus:
#       tf.config.experimental.set_memory_growth(gpu, True)
#     logical_gpus = tf.config.experimental.list_logical_devices('GPU')
#     print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
#   except RuntimeError as e:
#     # Memory growth must be set before GPUs have been initialized
#     print(e)

# variables
csv_name = "images.csv"
num_classes = 2
batch_size = 4
image_size = (512, 512)
validation_set_length = 128


# load image paths from csv file
def get_image_paths():
    return pd.read_csv(csv_name, header=None, index_col=0, squeeze=True).to_dict()


image_paths = get_image_paths()


class ImageHandler(keras.utils.Sequence):
    def __init__(self, batch_size, image_size, image_paths, size, do_augs):
        self.imageLoader = realtime_augmentation.ImageLoader(image_size, image_paths, do_augs)
        self.size = size

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        return self.imageLoader.get_random_batch(batch_size)


def unet(pretrained_weights=None, input_size=(512, 512, 1)):
    inputs = layers.Input(input_size)
    filter_num = 16
    conv1 = layers.Conv2D(filter_num * 1, 3, activation='relu', padding='same', kernel_initializer='he_normal')(inputs)
    conv1 = layers.Conv2D(filter_num * 1, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv1)
    batch1 = layers.BatchNormalization()(conv1)
    pool1 = layers.MaxPooling2D(pool_size=(2, 2))(batch1)

    conv2 = layers.Conv2D(filter_num * 2, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool1)
    conv2 = layers.Conv2D(filter_num * 2, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv2)
    batch2 = layers.BatchNormalization()(conv2)
    pool2 = layers.MaxPooling2D(pool_size=(2, 2))(batch2)

    conv3 = layers.Conv2D(filter_num * 4, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool2)
    conv3 = layers.Conv2D(filter_num * 4, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv3)
    batch3 = layers.BatchNormalization()(conv3)
    pool3 = layers.MaxPooling2D(pool_size=(2, 2))(batch3)

    conv4 = layers.Conv2D(filter_num * 8, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool3)
    conv4 = layers.Conv2D(filter_num * 8, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv4)
    batch4 = layers.BatchNormalization()(conv4)
    drop4 = layers.Dropout(0.5)(batch4)
    pool4 = layers.MaxPooling2D(pool_size=(2, 2))(drop4)

    conv5 = layers.Conv2D(filter_num * 16, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool4)
    conv5 = layers.Conv2D(filter_num * 16, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv5)
    batch5 = layers.BatchNormalization()(conv5)
    drop5 = layers.Dropout(0.5)(batch5)

    up6 = layers.Conv2D(filter_num * 8, 2, activation='relu', padding='same', kernel_initializer='he_normal')(
        layers.UpSampling2D(size=(2, 2))(drop5))
    merge6 = layers.concatenate([drop4, up6], axis=3)
    conv6 = layers.Conv2D(filter_num * 8, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge6)
    conv6 = layers.Conv2D(filter_num * 8, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv6)

    up7 = layers.Conv2D(filter_num * 4, 2, activation='relu', padding='same', kernel_initializer='he_normal')(
        layers.UpSampling2D(size=(2, 2))(conv6))
    merge7 = layers.concatenate([conv3, up7], axis=3)
    conv7 = layers.Conv2D(filter_num * 4, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge7)
    conv7 = layers.Conv2D(filter_num * 4, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv7)

    up8 = layers.Conv2D(filter_num * 2, 2, activation='relu', padding='same', kernel_initializer='he_normal')(
        layers.UpSampling2D(size=(2, 2))(conv7))
    merge8 = layers.concatenate([conv2, up8], axis=3)
    conv8 = layers.Conv2D(filter_num * 2, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge8)
    conv8 = layers.Conv2D(filter_num * 2, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv8)

    up9 = layers.Conv2D(filter_num * 1, 2, activation='relu', padding='same', kernel_initializer='he_normal')(
        layers.UpSampling2D(size=(2, 2))(conv8))
    merge9 = layers.concatenate([conv1, up9], axis=3)
    conv9 = layers.Conv2D(filter_num * 1, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge9)
    conv9 = layers.Conv2D(filter_num * 1, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv9)
    conv9 = layers.Conv2D(2, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv9)

    conv10 = layers.Conv2D(1, 1, activation='sigmoid')(conv9)

    model = Model(inputs, conv10)

    model.compile(optimizer=Adam(lr=1e-4), loss='binary_crossentropy', metrics=['accuracy'])
    # model.compile(optimizer=Adam(lr=5e-5), loss='binary_crossentropy', metrics=['accuracy'])
    # model.compile(optimizer=Adam(lr=1e-2), loss='binary_crossentropy', metrics=['accuracy'])

    model.summary()

    if (pretrained_weights):
        model.load_weights(pretrained_weights)

    return model


# expects an np array of size (width, height, 1)
def display_image(img):
    pImage.fromarray(np.uint8(img[:, :, 0] * 255), mode="L").show()


# split our image paths into training and validation image paths
validation_image_paths = dict(list(image_paths.items())[:validation_set_length])
train_image_paths = dict(list(image_paths.items())[validation_set_length:])


# create our image handler classes
train_images = ImageHandler(batch_size, image_size, train_image_paths, 2000, True)
validation_images = ImageHandler(batch_size, image_size, validation_image_paths, validation_set_length, False)

print("LEN  " + str(len(train_images)))

# create our model
# model = unet()
model = keras.models.load_model("trained_model")

callbacks = [
    keras.callbacks.ModelCheckpoint("xray_segmentation.h5", save_best_only=True)
]

# Train the model, doing validation at the end of each epoch.
epochs = 10
model.fit(train_images, epochs=epochs, validation_data=validation_images, callbacks=callbacks)

model.save("trained_model")

# model = keras.models.load_model('trained_model')

validation_predictions = model.predict(validation_images)
print(validation_predictions.shape)

combined_image = pImage.new("I", (image_size[0] * 2, image_size[1] * validation_set_length))
y = 0

for i in range(0, len(validation_image_paths)):
    image = pImage.open(list(validation_image_paths.keys())[i])
    image_handler = ImageHandler(1, image_size, {list(validation_image_paths.keys())[i] : list(validation_image_paths.values())[i]}, 1, False)
    prediction = model.predict(image_handler)
    predicted_image = pImage.fromarray(np.uint8(prediction[0][:, :, 0] * 255), mode="L")

    combined_image.paste(image, (0, y))
    combined_image.paste(predicted_image, (image_size[0], y))
    y += image_size[1]
    print("predicting validation image number " + str(y // image_size[1]))

combined_image.show()
# combined_image = pImage.fromarray(np.array(combined_image) * 255, mode="L")
combined_image.save("./validation_predictions " + str(datetime.datetime.now()).replace(":", "-").replace(".", "-") + ".png", "PNG")