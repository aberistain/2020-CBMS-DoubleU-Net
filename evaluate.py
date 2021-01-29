import os
from pathlib import Path
from glob import glob

import cv2
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import CustomObjectScope
from tqdm import tqdm

from train import tf_dataset
from utils import *


def read_image(x, resize=None):
    image = cv2.imread(x, cv2.IMREAD_COLOR)
    if resize:
        # resize (width, height)
        image = cv2.resize(image, resize, interpolation=cv2.INTER_AREA)
    image = np.clip(image - np.median(image) + 127, 0, 255)
    image = image / 255.0
    image = image.astype(np.float32)
    image = np.expand_dims(image, axis=0)
    return image


def read_mask(y, resize=None):
    mask = cv2.imread(y, cv2.IMREAD_GRAYSCALE)
    if resize:
        # resize (width, height)
        mask = cv2.resize(mask, resize, interpolation=cv2.INTER_AREA)
    mask = mask.astype(np.float32)
    mask = mask / 255.0
    mask = np.expand_dims(mask, axis=-1)
    return mask


def mask_to_3d(mask):
    mask = np.squeeze(mask)
    mask = [mask, mask, mask]
    mask = np.transpose(mask, (1, 2, 0))
    return mask


def parse(y_pred):
    y_pred = np.expand_dims(y_pred, axis=-1)
    y_pred = y_pred[..., -1]
    y_pred = y_pred.astype(np.float32)
    y_pred = np.expand_dims(y_pred, axis=-1)
    return y_pred


def evaluate_normal(model, x_data, y_data, model_image_size=None):
    THRESHOLD = 0.5
    total = []
    for i, (x, y) in tqdm(enumerate(zip(x_data, y_data)), total=len(x_data)):
        x = read_image(x, resize=model_image_size)
        y = read_mask(y, resize=model_image_size)
        _, h, w, _ = x.shape

        y_pred1 = parse(model.predict(x)[0][..., -2])
        y_pred2 = parse(model.predict(x)[0][..., -1])

        line = np.ones((h, 10, 3)) * 255.0

        all_images = [
            x[0] * 255.0, line,
            mask_to_3d(y) * 255.0, line,
            mask_to_3d(y_pred1) * 255.0
        ]
        mask = np.concatenate(all_images, axis=1)

        cv2.imwrite(f"results/{i}.png", mask)


smooth = 1.


def dice_coef(y_true, y_pred):
    y_true_f = tf.keras.layers.Flatten()(y_true)
    y_pred_f = tf.keras.layers.Flatten()(y_pred)
    intersection = tf.reduce_sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (tf.reduce_sum(y_true_f) + tf.reduce_sum(y_pred_f) + smooth)


def dice_loss(y_true, y_pred):
    return 1.0 - dice_coef(y_true, y_pred)


if __name__ == "__main__":
    np.random.seed(42)
    tf.random.set_seed(42)
    create_dir("results/")

    batch_size = 5

    test_path = str(Path(__file__).parent.joinpath('dataset', 'ISIC_2018'))
    model_path = str(Path(__file__).parent.joinpath('files', 'model.h5'))

    test_x = sorted(glob(os.path.join(test_path, "image", "*.jpg")))
    test_y = sorted(glob(os.path.join(test_path, "mask", "*.png")))
    test_dataset = tf_dataset(test_x, test_y, batch=batch_size)

    test_steps = (len(test_x) // batch_size)
    if len(test_x) % batch_size != 0:
        test_steps += 1

    model = load_model_weight(model_path)
    model.evaluate(test_dataset, steps=test_steps)
    evaluate_normal(model, test_x, test_y, model_image_size=(512, 384))
