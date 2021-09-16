import copy
import os
import sys
from collections import OrderedDict

import numpy as np
import SimpleITK as sitk
from batchgenerators.utilities.file_and_folder_operations import *

from pymic.io.image_read_write import (load_image_as_nd_array,
                                          save_array_as_nifty_volume,
                                          save_nd_array_as_image)
from pymic.util.image_process import (
    convert_label, crop_ND_volume_with_bounding_box, get_ND_bounding_box,
    set_ND_volume_roi_with_bounding_box_range)
from path_config import path_dict
if __name__ == "__main__":
    if(len(sys.argv) < 3):
        print('Number of arguments should be 3. e.g.')
        print('   python get_final_test.py input_dir output_dir')
        exit()
    input_dir  = str(sys.argv[1])
    output_dir = str(sys.argv[2])
    #first_crop = "/mnt/data1/swzhai/dataset/MyoPS/data_preprocessed/imagesTs/crop_information.json"
    #second_crop = "/mnt/data1/swzhai/dataset/MyoPS/nnUNet_raw_data_base/nnUNet_raw_data/Task112_MyoPS/imagesTs/crop_information.json"
    first_crop  = path_dict['MyoPS_data_dir']  + "/data_preprocessed/imagesTs/crop_information.json"
    second_crop = path_dict['nnunet_raw_data_dir'] + "/Task112_MyoPS/imagesTs/crop_information.json"
    first_json  = load_json((first_crop))
    second_json = load_json(second_crop)

    # seg_dir  = "../MyoPS-main/myops/result_test/nnunet_2d"
    # save_dir = "../MyoPS-main/myops/result_test/test_ensemble_original"
    img_dir = path_dict['MyoPS_data_dir'] + "/data_raw/imagesTs"
    maybe_mkdir_p(output_dir)
    for pred_name in second_json.keys():
        seg_name = pred_name.split("/")[-1]
        print(seg_name)
        save_name = seg_name
        img_name = seg_name.replace(".nii.gz", "_C0.nii.gz")
        seg_array = load_image_as_nd_array(join(input_dir, seg_name))["data_array"].squeeze(0)
        img_array = load_image_as_nd_array(join(img_dir, img_name))["data_array"].squeeze(0)
        zero_array = np.zeros_like(img_array)
        print(seg_array.shape, zero_array.shape)
        first_json_bbox = first_json[join(img_dir, img_name)]
        second_json_bbox = second_json[pred_name]
        print(first_json_bbox)
        print(second_json_bbox)
        # for D, H, W
        crop_bbox_min = [0, 0, 0]
        crop_bbox_max = [0, 0, 0]
        for i in range(3): 
            crop_bbox_min[i] = first_json_bbox["crop_bbox_min"][i] + second_json_bbox["crop_bbox_min"][i]
            crop_bbox_max[i] = first_json_bbox["crop_bbox_min"][i] + second_json_bbox["crop_bbox_max"][i]
        print(crop_bbox_min, crop_bbox_max, seg_array.shape)
        output_seg_array = set_ND_volume_roi_with_bounding_box_range(zero_array, crop_bbox_min, crop_bbox_max, seg_array, False)
        output_seg_array = convert_label(output_seg_array, [1, 2, 3, 4, 5], [200, 500, 600, 1220, 2221])
        save_nd_array_as_image(output_seg_array, join(output_dir, save_name), join(img_dir, img_name))
