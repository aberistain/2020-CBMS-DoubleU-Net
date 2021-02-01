import argparse
from glob import glob
from pathlib import Path
from abc import ABC, abstractmethod

import cv2
import numpy as np
import tensorflow as tf
from tqdm import tqdm

from evaluate import mask_to_3d, parse
from utils import create_dir, load_model_weight


class Inference(ABC):
    def __init__(self, model_path):
        self.model_path = model_path
        self.model = self._load_model(model_path=self.model_path)

    @abstractmethod
    def _load_model(self, model_path):
        raise NotImplementedError()

    @abstractmethod
    def _infer_image(self, image):
        raise NotImplementedError()

    def _read_image(self, x, resize=None):
        image = cv2.imread(x, cv2.IMREAD_COLOR)
        original_size = (image.shape[1], image.shape[0])  # (width, height)
        if resize:
            # resize (width, height)
            image = cv2.resize(image, resize, interpolation=cv2.INTER_AREA)
        image = np.clip(image - np.median(image) + 127, 0, 255)
        image = image / 255.0
        image = image.astype(np.float32)
        image = np.expand_dims(image, axis=0)
        return image, original_size

    def infer_image_file_path(self, image_file_path, resize=None):
        image, img_original_shape = self._read_image(x=image_file_path, resize=resize)
        out_image = self._infer_image(image)
        return cv2.resize(mask_to_3d(out_image) * 255.0, img_original_shape, interpolation=cv2.INTER_AREA),\
               img_original_shape


class InferenceFullModel(Inference):
    def __init__(self, model_path):
        super().__init__(model_path)

    def _load_model(self, model_path):
        return load_model_weight(model_path)

    def _infer_image(self, image):
        return parse(self.model.predict(image)[0][..., -2])


class InferenceTFLite(Inference):
    def __init__(self, model_path):
        super().__init__(model_path)

    def _load_model(self, model_path):
        interpreter = tf.lite.Interpreter(model_path=model_path)
        interpreter.allocate_tensors()
        return interpreter

    def _infer_image(self, image):
        input_details = self.model.get_input_details()
        output_details = self.model.get_output_details()
        # input_shape = input_details[0]['shape']
        input_data = np.array(image, dtype=np.float32)
        self.model.set_tensor(input_details[0]['index'], input_data)
        self.model.invoke()
        # The function `get_tensor()` returns a copy of the tensor data.
        # Use `tensor()` in order to get a pointer to the tensor.
        return self.model.get_tensor(output_details[0]['index'])[0][..., -2]

def process_images(model, image_path_list, out_path, model_image_size=None):
    for i, im_path in tqdm(enumerate(image_path_list), total=len(image_path_list)):
        out_image, img_original_shape = model.infer_image_file_path(im_path, resize=model_image_size)
        cv2.imwrite(str(Path(out_path, f'{Path(im_path).stem}.png')), out_image)


def get_image_path_list_from_file(file_path: Path):
    image_file_path_list = list()
    if file_path.is_file():
        with file_path.open(mode='r') as f:
            for line in f:
                file_path = Path(line)
                if file_path.exists() and file_path.is_file():
                    if file_path.suffix in ('.jpg', '.jpeg', '.png', '.bmp'):
                        image_file_path_list.append(line)


def generate_image_file_list(file_path: Path):
    out_path = Path('file_list.txt')
    if file_path.is_dir():
        with out_path.open(mode='w') as f:
            for currentFile in file_path.iterdir():
                f.write(f'{currentFile}')


def convert_model_to_tf_lite(model_path=str(Path(__file__).parent.joinpath('files', 'model.h5')),
                             output_model_path='model.tflite'):
    model = load_model_weight(model_path)
    converter = tf.lite.TFLiteConverter.from_keras_model(model=model)
    tflite_model = converter.convert()
    # Save the model.
    with open(output_model_path, 'wb') as f:
        f.write(tflite_model)


def main():
    # parser = argparse.ArgumentParser(
    #     description="Lesion segmentation from paper:"
    #                 "@INPROCEEDINGS{9183321, "
    #                 "author={D. {Jha} and M. A. {Riegler} and D. {Johansen} and P. {Halvorsen} and H. D. {Johansen}}, "
    #                 "booktitle={2020 IEEE 33rd International Symposium on Computer-Based Medical Systems (CBMS)}, "
    #                 "title={DoubleU-Net: A Deep Convolutional Neural Network for Medical Image Segmentation}, "
    #                 "year={2020},"
    #                 "pages={558-564}}")
    #
    # # Required positional argument
    # parser.add_argument('input_file', type=str,
    #                     help='Path to txt file containing one image file path per line')

    out_path = str(Path(__file__).parent.joinpath('results'))
    create_dir(out_path)

    test_path = str(Path(__file__).parent.joinpath('dataset', 'ISIC_2018'))
    model_path = str(Path(__file__).parent.joinpath('files', 'model.h5'))
    model_path_tf_lite = str(Path(__file__).parent.joinpath('files', 'model.tflite'))

    test_x = sorted(glob(str(Path(test_path, "image", "*.jpg"))))
    infering_model = InferenceTFLite(model_path=model_path_tf_lite)
    process_images(model=infering_model, image_path_list=test_x, out_path=out_path, model_image_size=(512, 384))


if __name__ == "__main__":
    main()

