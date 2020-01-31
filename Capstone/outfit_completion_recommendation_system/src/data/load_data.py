from skimage import io, exposure
import h5py
import numpy as np
from os import listdir, path, mkdir
from tqdm import tqdm
from utils.utils_data_download import DataDownloadUtilities
from utils.utils_data_preprocessing import DataPreprocessingUtilities


class LoadData:

    TRAINING_DATA_PCNT = 0.8
    VALIDATION_DATA_PCNT = 0.1
    TEST_DATA_PCNT = 0.1

    def __init__(self, image_cat):
        self.image_cat = image_cat
        self.set_paths(
            base_file_path="data/external/",
            raw_data_save_path="data/raw/",
            interim_data_save_path="data/interim/",
            train_test_data_save_path="data/processed/",
        )
        self.image_retrieval()
        self.process_all_images()

    def set_paths(
        self,
        base_file_path,
        raw_data_save_path,
        interim_data_save_path,
        train_test_data_save_path,
    ):

        self.base_file_path = base_file_path
        self.raw_data_save_path = raw_data_save_path
        self.interim_data_save_path = interim_data_save_path
        self.train_test_data_save_path = train_test_data_save_path

    @staticmethod
    def add_directory(base_path, image_cat, image_type):
        """add directory if the directory does not exist"""
        if not path.exists(base_path + "/{}_{}".format(image_cat, image_type)):
            mkdir(base_path + "/{}_{}".format(image_cat, image_type))

    def image_retrieval(self):
        """ Retrieve all image data and save all images to the save path specified"""
        image_raw_mapping = DataDownloadUtilities().get_metadata_list(
            self.base_file_path, self.image_cat
        )

        for image_type in ["product", "scene"]:
            self.add_directory(self.base_file_path, self.image_cat, image_type)
            # remove duplicates
            image_raw_list = [i[image_type] for i in image_raw_mapping]
            image_raw_list_unique = list(dict.fromkeys(image_raw_list))
            output_img_path = self.raw_data_save_path + "/{}_{}".format(
                self.image_cat, image_type
            )

            # get all unique images that does not exist yet
            existed = [x[:-4] for x in listdir(output_img_path)]
            image_raw_list_unique_new = list(set(image_raw_list_unique) - set(existed))

            # get all the images that hasn't been downloaded
            if len(image_raw_list_unique_new) > 0:
                print("Downloading {} data ...".format(image_type))
                for signature in tqdm(image_raw_list_unique_new):
                    image_url = DataDownloadUtilities.convert_to_url(signature)
                    try:
                        img = io.imread(image_url)
                        io.imsave(output_img_path + "/{}.jpg".format(signature), img)
                    except:
                        print(image_url)
                print("{} new images retrieved".format(len(image_raw_list_unique_new)))

            else:
                print("no new images detected, using archived images")

    def process_all_images(self):
        """ Crop all the images based on the bounding box and resize them to 224 x 224"""
        # add directories
        for image_type in ["product", "scene"]:
            self.add_directory(self.base_file_path, self.image_cat, image_type)

        raw_scene_data_path = self.raw_data_save_path + "/{}_{}".format(
            self.image_cat, "scene"
        )

        raw_product_data_path = self.raw_data_save_path + "/{}_{}".format(
            self.image_cat, "product"
        )

        interim_scene_data_save_path = self.interim_data_save_path + "/{}_{}".format(
            self.image_cat, "scene"
        )

        interim_product_data_save_path = self.interim_data_save_path + "/{}_{}".format(
            self.image_cat, "product"
        )

        if len(listdir(interim_scene_data_save_path)) == 0:
            # get all the product mapping
            product_scene_mapping_unprocessed = DataDownloadUtilities().get_metadata_list(
                self.base_file_path, self.image_cat
            )
            product_scene_mapping_processed = []

            # for all scenes
            for image_name in listdir(raw_scene_data_path):
                # for all image mappings
                for i, mapping in enumerate(product_scene_mapping_unprocessed):
                    # find the corresponding scene from fashion_scene_mapping
                    if mapping["scene"] == image_name[:-4]:
                        # remove the mapping from the unprocessed list
                        product_scene_mapping_processed.append(
                            product_scene_mapping_unprocessed.pop(i)
                        )
                        image = io.imread(raw_scene_data_path + "/" + image_name)
                        # get cropped image
                        cropped_image = DataPreprocessingUtilities.crop_image_bounding_box(
                            image, mapping["bbox"]
                        )
                        # save the qualified cropped image
                        if (cropped_image is not None) and (
                            exposure.is_low_contrast(cropped_image) == 0
                        ):
                            io.imsave(
                                interim_scene_data_save_path
                                + "/{}_{}".format(mapping["product"], image_name),
                                cropped_image,
                            )

        if len(listdir(interim_product_data_save_path)) == 0:
            # copy all the product image to interim directory
            DataPreprocessingUtilities.copy_directory(
                raw_product_data_path, interim_product_data_save_path
            )

        print("all {} images processed".format(self.image_cat))

    def create_train_test_validation_data(self):
        """80-10-10 train-validation-test split"""
        product_scene_mapping = DataDownloadUtilities().get_metadata_list(
            self.base_file_path, self.image_cat
        )

        # shuffle product_scene_mapping
        np.random.shuffle(product_scene_mapping)

        # generate training dataset
        training_split_point = int(len(product_scene_mapping) * self.TRAINING_DATA_PCNT)
        training_data_mapping = np.array(product_scene_mapping)[:training_split_point]

        # generate validation dataset
        validation_split_point = (
            int(len(product_scene_mapping) * self.TRAINING_DATA_PCNT)
            + training_split_point
        )
        validation_data_mapping = np.array(product_scene_mapping)[
            training_split_point:validation_split_point
        ]

        # generate test dataset
        test_data_mapping = np.array(product_scene_mapping)[validation_split_point:]


if __name__ == "__main__":
    LoadData("fashion").create_train_test_validation_data()
