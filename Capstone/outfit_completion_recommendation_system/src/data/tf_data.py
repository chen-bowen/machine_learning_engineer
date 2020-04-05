import tensorflow as tf
import os
import numpy as np


class TFDataset:

    MODEL_IMAGE_SIZE = 224
    NUM_AUGUMENT_SAMPLES = 8

    def __init__(self, image_cat, product_type):

        self.image_cat = image_cat
        self.product_type = product_type
        self.set_paths(train_test_data_save_path="data/processed/")

    def set_paths(self, train_test_data_save_path):
        """Set all the base paths for """
        self.train_test_data_save_path = train_test_data_save_path

    @staticmethod
    def rotate(x: tf.Tensor) -> tf.Tensor:
        """Rotation augmentation

        Args:
            x: Image

        Returns:
            Augmented image
        """

        # Rotate 0, 90, 180, 270 degrees
        return tf.image.rot90(
            x, tf.random.uniform(shape=[], minval=0, maxval=4, dtype=tf.int32)
        )

    @staticmethod
    def flip(x: tf.Tensor) -> tf.Tensor:
        """Flip augmentation

        Args:
            x: Image to flip

        Returns:
            Augmented image
        """
        x = tf.image.random_flip_left_right(x)
        x = tf.image.random_flip_up_down(x)

        return x

    @staticmethod
    def color(x: tf.Tensor) -> tf.Tensor:
        """Color augmentation

        Args:
            x: Image

        Returns:
            Augmented image
        """
        x = tf.image.random_hue(x, 0.08)
        x = tf.image.random_saturation(x, 0.6, 1.6)
        x = tf.image.random_brightness(x, 0.05)
        x = tf.image.random_contrast(x, 0.7, 1.3)

        return x

    @staticmethod
    def zoom(x: tf.Tensor) -> tf.Tensor:
        """Zoom augmentation

        Args:
            x: Image

        Returns:
            Augmented image
        """

        # Generate 20 crop settings, ranging from a 1% to 20% crop.
        scales = list(np.arange(0.8, 1.0, 0.01))
        boxes = np.zeros((len(scales), 4))

        for i, scale in enumerate(scales):
            x1 = y1 = 0.5 - (0.5 * scale)
            x2 = y2 = 0.5 + (0.5 * scale)
            boxes[i] = [x1, y1, x2, y2]

        def random_crop(img):
            # Create different crops for an image
            crops = tf.image.crop_and_resize(
                [img],
                boxes=boxes,
                box_indices=np.zeros(len(scales)),
                crop_size=(MODEL_IMAGE_SIZE, MODEL_IMAGE_SIZE),
            )
            # Return a random crop
            return crops[
                tf.random.uniform(shape=[], minval=0, maxval=len(scales), dtype=tf.int32)
            ]

        choice = tf.random.uniform(shape=[], minval=0.0, maxval=1.0, dtype=tf.float32)

        # Only apply cropping 50% of the time
        return tf.cond(choice < 0.5, lambda: x, lambda: random_crop(x))

    @staticmethod
    def get_label(file_path):
        # convert the path to a list of path components
        parts = tf.strings.split(file_path, os.path.sep)
        # string before .jpg, after _ is the label
        file_name = tf.strings.split(parts[-1], "_")
        label_name = tf.strings.split(file_name[1], ".")[0]
        return label_name

    @staticmethod
    def decode_img(img):
        # convert the compressed string to a 3D uint8 tensor
        img = tf.image.decode_jpeg(img, channels=3)
        # Use `convert_image_dtype` to convert to floats in the [0,1] range.
        img = tf.image.convert_image_dtype(img, tf.float32)
        # resize the image to the desired size.
        return img

    @staticmethod
    def process_image(file_path, data_type):
        label = TFDataset.get_label(file_path)
        # load the raw data from the file as a string
        img = tf.io.read_file(file_path)
        img = TFDataset.decode_img(img)
        augmentations = [
            TFDataset.flip,
            TFDataset.color,
            TFDataset.zoom,
            TFDataset.rotate,
        ]

        # random add data augumentations
        if data_type == "training":
            applied_f = augmentations[np.random.choice(len(augmentations), 1)[0]]
            img = tf.cond(
                tf.random.uniform([], 0, 1) > 0.75, lambda: applied_f(img), lambda: img
            )
            return img, label

        return img

    def get_tf_dataset(self, data_type):
        """create tensorflow dataset for both label image and data image, 
            given what type of data is"""
        data_path = (
            self.train_test_data_save_path
            + "/"
            + data_type
            + "/{}_{}/*".format(self.image_cat, self.product_type)
        )

        list_data_dir = tf.data.Dataset.list_files(str(data_path)).repeat(
            TFDataset.NUM_AUGUMENT_SAMPLES
        )
        labeled_data = list_data_dir.map(
            lambda record: self.process_image(record, data_type),
            num_parallel_calls=tf.data.experimental.AUTOTUNE,
        )
        return labeled_data
