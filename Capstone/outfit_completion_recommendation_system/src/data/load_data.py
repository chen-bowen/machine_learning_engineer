from skimage import io
import h5py
from os import listdir
from tqdm import tqdm 
from utils.utils_data_download import DataDownloadUtilities 

class LoadData:

    def __init__(self,
                 base_file_path, 
                 data_save_path):
    
        self.base_file_path = base_file_path
        self.data_save_path = data_save_path

    def image_retrieval(self, image_cat= "fashion", image_type="product"):
        """ Retrieve all image data and save all images to the save path specified"""
        image_raw_mapping = DataDownloadUtilities().get_metadata_list(self.base_file_path, image_cat)

        # remove duplicates
        image_raw_list = [i[image_type] for i in image_raw_mapping]
        image_raw_list_unique = list(dict.fromkeys(image_raw_list))
        output_img_path = self.data_save_path + "/{}_{}".format(image_cat, image_type)

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

if __name__ == "__main__":
    LoadData("data/external/", "data/raw/").image_retrieval()