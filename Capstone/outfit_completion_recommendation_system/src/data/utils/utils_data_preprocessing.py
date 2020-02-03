from skimage import io
from cv2 import resize, INTER_CUBIC
import numpy as np
from os import listdir


class DataPreprocessingUtilities:

    MODEL_IMAGE_SIZE = 224

    def __init__(self):
        pass

    @staticmethod
    def crop_image_bounding_box(image, bounding_box):
        """ Find the largest cropped image by cropping from left, right, top, bottom"""

        left, top, right, bottom = bounding_box

        try:
            h, w, _ = image.shape

            # enlarge the bounding box by 5% and get the image crop point
            left = max(0, w * left * (1 - 0.025))
            top = max(0, h * top * (1 - 0.025))
            right = min(w * right * (1 + 0.025), w - 1)
            bottom = min(h * bottom * (1 + 0.025), h - 1)

            # crop images
            left_crop = image[:, : round(left), :]
            right_crop = image[:, round(right) :, :]
            top_crop = image[: round(top), :, :]
            bottom_crop = image[round(bottom) :, :, :]

        except ValueError:

            h, w = image.shape

            # enlarge the bounding box by 5% and get the image crop point
            left = max(0, w * left * (1 - 0.025))
            top = max(0, h * top * (1 - 0.025))
            right = min(w * right * (1 + 0.025), w - 1)
            bottom = min(h * bottom * (1 + 0.025), h - 1)

            # crop images
            left_crop = image[:, : round(left)]
            right_crop = image[:, round(right) :]
            top_crop = image[: round(top), :]
            bottom_crop = image[round(bottom) :, :]

        # find the largest area
        cropped_list = [left_crop, top_crop, right_crop, bottom_crop]
        largest_area_index = np.argmax([np.product(i.shape) for i in cropped_list])
        cropped_image = cropped_list[largest_area_index]

        # if the cropped image is less than 1/5 of original size
        if np.product(cropped_image.shape) < 1 / 5 * np.product(image.shape):
            return None

        # reshape the image to 224 x 224
        cropped_image = resize(
            cropped_image,
            (
                DataPreprocessingUtilities.MODEL_IMAGE_SIZE,
                DataPreprocessingUtilities.MODEL_IMAGE_SIZE,
            ),
            interpolation=INTER_CUBIC,
        )

        return cropped_image

    @staticmethod
    def copy_directory(original_dir, destination_dir):
        """ Copy all files from original directory to destination"""
        for filename in listdir(original_dir):
            image = io.imread(original_dir + "/" + filename)
            io.imsave(destination_dir + "/" + filename, image)
