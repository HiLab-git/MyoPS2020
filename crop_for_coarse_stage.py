import os
import numpy as np
import SimpleITK as sitk
from collections import OrderedDict
from batchgenerators.utilities.file_and_folder_operations import save_json, maybe_mkdir_p
from pymic.util.image_process import get_ND_bounding_box, crop_ND_volume_with_bounding_box, \
    set_ND_volume_roi_with_bounding_box_range, convert_label
from pymic.io.image_read_write import save_array_as_nifty_volume
from path_config import path_dict

def crop_dataset_with_bbox(input_dir, output_dir, crop_bbox_min_hw, crop_bbox_max_hw, label_convert=False):
    images_name_list = os.listdir(input_dir)
    images_name_list.sort()
    json_dict = OrderedDict()
    for image_name in images_name_list:
        image_path = os.path.join(input_dir, image_name)
        print(image_path)
        image_sitk = sitk.ReadImage(image_path)
        image_npy = sitk.GetArrayFromImage(image_sitk)
        image_shape = image_npy.shape
        # do not crop along depth dimension
        crop_bbox_min_new = [0] + crop_bbox_min_hw
        crop_bbox_max_new = [image_shape[0]] + crop_bbox_max_hw
        # avoid out of indexes
        crop_bbox_min_new[1] = max(crop_bbox_min_new[1], 0)
        crop_bbox_min_new[2] = max(crop_bbox_min_new[2], 0)
        crop_bbox_max_new[1] = min(crop_bbox_max_new[1], image_shape[1])
        crop_bbox_max_new[2] = min(crop_bbox_max_new[2], image_shape[2])
        print(crop_bbox_min_new, crop_bbox_max_new)
        json_dict[image_path] = {"crop_bbox_min": crop_bbox_min_new, "crop_bbox_max": crop_bbox_max_new}
        image_output_npy = crop_ND_volume_with_bounding_box(image_npy, crop_bbox_min_new, crop_bbox_max_new)
        if label_convert:
            image_output_npy = convert_label(image_output_npy, source_list, target_list)
        save_array_as_nifty_volume(image_output_npy, os.path.join(output_dir, image_name), image_path)
    save_json(json_dict, os.path.join(output_dir, "crop_information.json"))

        

if __name__ == "__main__":
    # "/home/guotai/disk2t/data/heart/MyoPS/data_raw"
    input_dir  = path_dict["MyoPS_data_dir"]  + "/data_raw"
    output_dir = path_dict["MyoPS_data_dir"]  + "/data_preprocessed"
    if(not os.path.isdir(output_dir)):
        os.mkdir(output_dir)
    source_list = [200, 500, 600, 1220, 2221]
    target_list = [1, 2, 3, 4, 5]
    imagesTr_dir = os.path.join(input_dir, "imagesTr")
    labelsTr_dir = os.path.join(input_dir, "labelsTr")
    imagesTs_dir = os.path.join(input_dir, "imagesTs")
    labelsTr_name_list = os.listdir(labelsTr_dir)
    labelsTr_name_list.sort()
    num_data = len(labelsTr_name_list)
    imagesTr_output_dir = os.path.join(output_dir, "imagesTr")
    maybe_mkdir_p(imagesTr_output_dir)
    labelsTr_output_dir = os.path.join(output_dir, "labelsTr")
    maybe_mkdir_p(labelsTr_output_dir)
    imagesTs_output_dir = os.path.join(output_dir, "imagesTs")
    maybe_mkdir_p(imagesTs_output_dir)

    crop_bbox_min_all = []
    crop_bbox_max_all = []
    margin = [0, 30, 30]
    for i in range(num_data):
        labelTr_name = labelsTr_name_list[i]
        labelTr_path = os.path.join(labelsTr_dir, labelTr_name)
        # print(i + 1, labelTr_path)
        labelTr_sitk = sitk.ReadImage(labelTr_path)
        labelTr_npy = sitk.GetArrayFromImage(labelTr_sitk)
        crop_bbox_min, crop_bbox_max = get_ND_bounding_box(labelTr_npy, margin=margin)
        # print(crop_bbox_min, crop_bbox_max)
        crop_bbox_min_all.append(crop_bbox_min)
        crop_bbox_max_all.append(crop_bbox_max)
    # [:, 1:] means that we only count the range of indexes along height and width dimensions
    crop_bbox_min_hw = np.min(np.array(crop_bbox_min_all)[:, 1:], axis=0).tolist()
    crop_bbox_max_hw = np.max(np.array(crop_bbox_max_all)[:, 1:], axis=0).tolist()
    print("the minimum of bounding box along height and width dimensions are: {0:}\nthe maximum of bounding box along"
            " height and width dimensions are: {1:}".format(crop_bbox_min_hw, crop_bbox_max_hw))
    crop_dataset_with_bbox(imagesTr_dir, imagesTr_output_dir, crop_bbox_min_hw, crop_bbox_max_hw)
    crop_dataset_with_bbox(labelsTr_dir, labelsTr_output_dir, crop_bbox_min_hw, crop_bbox_max_hw, label_convert=True)
    crop_dataset_with_bbox(imagesTs_dir, imagesTs_output_dir, crop_bbox_min_hw, crop_bbox_max_hw)
