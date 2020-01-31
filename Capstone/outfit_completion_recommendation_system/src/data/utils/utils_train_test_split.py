from skimage import io
from cv2 import resize, INTER_CUBIC
import numpy as np
from os import listdir
from tqdm import tqdm
import warnings

warnings.filterwarnings("ignore")


class TrainTestSplitUtilities:
    def __init__(self):
        pass

    def copy_image_to_dir(
        product_scene_mapping,
        image_cat,
        all_scene_images,
        all_product_images,
        original_dir,
        new_dir,
    ):
        """Move all images mentioned in base file to new dir,
          return list of processed images"""

        all_scene_images_processed = listdir(
            new_dir + "/{}_{}".format(image_cat, "scene")
        )
        all_product_images_processed = listdir(
            new_dir + "/{}_{}".format(image_cat, "product")
        )

        print("Creating {} {} data".format(image_cat, new_dir.split("/")[-1]))

        for mapping in tqdm(product_scene_mapping):
            # loop through all product-scene mappings, look for matching image
            scene_signature = mapping["scene"]
            product_singature = mapping["product"]

            for i, image_name in enumerate(all_scene_images):
                # copy scene images that are in the product_scene_mapping to new destination
                if (
                    (scene_signature == image_name.split("_")[1][:-4])
                    and (product_singature == image_name.split("_")[0])
                    and (image_name not in all_scene_images_processed)
                ):
                    all_scene_images_processed.append(image_name)
                    image = io.imread(
                        original_dir + "/{}_{}/".format(image_cat, "scene") + image_name
                    )
                    io.imsave(
                        new_dir + "/{}_{}/".format(image_cat, "scene") + image_name,
                        image,
                    )

            for i, image_name in enumerate(all_product_images):
                # copy product images that are in the product_scene_mapping to new destination
                if (product_singature == image_name[:-4]) and (
                    image_name not in all_product_images_processed
                ):

                    all_product_images_processed.append(image_name)
                    image = io.imread(
                        original_dir
                        + "/{}_{}/".format(image_cat, "product")
                        + image_name
                    )
                    io.imsave(
                        new_dir + "/{}_{}/".format(image_cat, "product") + image_name,
                        image,
                    )

        return all_scene_images_processed, all_product_images_processed
