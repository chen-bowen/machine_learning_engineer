from skimage import io, exposure
import h5py
from os import listdir
from tqdm import tqdm 
from utils.utils_data_download import DataDownloadUtilities 
from utils.utils_data_preprocessing import DataPreprocessingUtilities

class LoadData:

    def __init__(self,
                 base_file_path, 
                 raw_data_save_path,
                 processed_data_save_path):
    
        self.base_file_path = base_file_path
        self.raw_data_save_path = raw_data_save_path
        self.processed_data_save_path = processed_data_save_path
    


    def image_retrieval(self, image_cat= "fashion", image_type="product"):
        """ Retrieve all image data and save all images to the save path specified"""
        image_raw_mapping = DataDownloadUtilities().get_metadata_list(self.base_file_path, image_cat)

        # remove duplicates
        image_raw_list = [i[image_type] for i in image_raw_mapping]
        image_raw_list_unique = list(dict.fromkeys(image_raw_list))
        output_img_path = self.raw_data_save_path + "/{}_{}".format(image_cat, image_type)

        # get all unique images that does not exist yet
        existed = [x[:-4] for x in listdir(output_img_path)]
        image_raw_list_unique_new = list(set(image_raw_list_unique) - set(existed))
        
        if len(image_raw_list_unique_new) > 0:
            print("Downloading data ...")
            for signature in tqdm(image_raw_list_unique_new):
                image_url = DataDownloadUtilities.convert_to_url(signature)
                try:
                    img = io.imread(image_url)
                    io.imsave(output_img_path + '/{}.jpg'.format(signature), img)
                except:
                    print(image_url)
            print("{} new images retrieved".format(len(image_raw_list_unique_new)))

        else:
            print("no new images detected, using archived images")
        
    def crop_all_images(self, image_cat= "fashion", image_type="product"):
        """ Crop all the images based on the bounding box and resize them to 224 x 224"""
        fashion_product_scene_mapping_unprocessed = DataDownloadUtilities().get_metadata_list(self.base_file_path, image_cat)
        processed_data_save_path = self.processed_data_save_path + "/{}_{}".format(image_cat, "scene")
        raw_data_path = self.raw_data_save_path + "/{}_{}".format(image_cat, "scene")
        fashion_product_scene_mapping_processed = []

        # for all scenes
        for image_name in listdir(raw_data_path):
            # for all image mappings
            for i, mapping in enumerate(fashion_product_scene_mapping_unprocessed):
                # find the corresponding scene from fashion_scene_mapping
                if mapping["scene"] == image_name[:-4]:
                    # remove the mapping from the unprocessed list
                    fashion_product_scene_mapping_processed.append(fashion_product_scene_mapping_unprocessed.pop(i))
                    image = io.imread(raw_data_path + "/" + image_name)
                    # get cropped image
                    cropped_image = DataPreprocessingUtilities.crop_image_bounding_box(image, mapping["bbox"])
                    # save the qualified cropped image
                    if (cropped_image is not None) and (exposure.is_low_contrast(cropped_image) == 0):
                        io.imsave(processed_data_save_path + '/{}_{}'.format(mapping["product"], image_name), cropped_image)


if __name__ == "__main__":
    LoadData("data/external/", "data/raw/",  "data/processed/").crop_all_images()