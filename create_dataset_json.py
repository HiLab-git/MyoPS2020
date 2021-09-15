import json
import os
from collections import OrderedDict
from batchgenerators.utilities.file_and_folder_operations import save_json
import shutil
from pymic.io.image_read_write import load_image_as_nd_array, save_nd_array_as_image
from pymic.util.image_process import convert_label
from path_config import path_dict

if __name__ == "__main__":
    nnUNet_data_dir = path_dict['nnunet_raw_data_dir'] + "/Task112_MyoPS"
    lablesTr_list = (os.listdir(os.path.join(nnUNet_data_dir, "labelsTr")))
    lablesTr_list.remove("crop_information.json")
    lablesTr_list.sort()
    imagesTs_list = (os.listdir(os.path.join(nnUNet_data_dir, "imagesTs")))
    imagesTs_list.remove("crop_information.json")
    imagesTs_list.sort()
    tmp_list = []
    for i, item in enumerate(imagesTs_list):
        if "0000" in item:
            item = item.replace("_0000", "")
            tmp_list.append(item)
            print(item)
        else:
            continue
    imagesTs_list = tmp_list
    json_dict = OrderedDict()
    json_dict['name'] = "MyoPS"
    json_dict['description'] = "Myocardial pathology segmentation combining multi-sequence CMR"
    json_dict['tensorImageSize'] = "4D"
    json_dict['reference'] = "TBC"
    json_dict['licence'] = "CC BY-NC-ND"
    json_dict['release'] = "0.0"
    json_dict['modality'] = {
        "0": "C0",
        "1": "DE",
        "2": "T2",
        "3": "coarse_seg"
    }
    json_dict['labels'] = {
        "0": "background",
        "1": "myo",
        "2": "lv",
        "3": "rv",
        "4": "edema",
        "5": "scars"
    }
    json_dict['numTraining'] = len(lablesTr_list)
    json_dict['numTest'] = len(imagesTs_list)
    json_dict['training'] = [{'image': "./imagesTr/{0:}".format(item), "label": "./labelsTr/{0:}".format(item)} for item in
                                lablesTr_list]
    json_dict['test'] = ["./imagesTs/{0:}".format(item) for item in imagesTs_list]

    save_json(json_dict, os.path.join(nnUNet_data_dir, "dataset.json"))
