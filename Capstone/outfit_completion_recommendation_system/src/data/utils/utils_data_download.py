import json

class DataDownloadUtilities:

    def __init__(self):
        pass

    @staticmethod
    def convert_to_url(signature):
        prefix = 'http://i.pinimg.com/400x/%s/%s/%s/%s.jpg'
        return prefix % (signature[0:2], signature[2:4], signature[4:6], signature)
   
    @staticmethod
    def get_metadata_list(base_file_path, filename):
        with open(base_file_path + filename + ".json") as json_file:
            product_scene_mapping = [json.loads(line) for line in json_file]
        return product_scene_mapping