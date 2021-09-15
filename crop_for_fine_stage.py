import os
import shutil
import sys
from collections import OrderedDict

import SimpleITK as sitk
from batchgenerators.utilities.file_and_folder_operations import (load_json,
                                                                  load_pickle,
                                                                  save_json)
from numpy.lib.npyio import save

from pymic.io.image_read_write import save_array_as_nifty_volume
from pymic.util.image_process import (convert_label,
                                      crop_ND_volume_with_bounding_box,
                                      get_ND_bounding_box)
from path_config import path_dict

def find(name, path):
    for root, dirs, files in os.walk(path):
        if name in files:
            return os.path.join(root, name)

def move_file(origin_path,moved_path):
    dir_files=os.listdir(origin_path)            #得到该文件夹下所有的文件
    for file in  dir_files:
        file_path=os.path.join(origin_path,file)   #路径拼接成绝对路径
        if os.path.isfile(file_path):           #如果是文件，就打印这个文件路径
            if file.endswith(".txt"):
                if os.path.exists(os.path.join(moved_path, file)):
                    print("有重复文件！！，跳过，不移动！！！")
                    continue
                else:
                    shutil.move(file_path, moved_path)
        if os.path.isdir(file_path):  #如果目录，就递归子目录
            move_file(file_path,moved_path)
    print("移动文件成功！")

def crop_acc_mask(images_dir, images_output_dir, masks_dir, mask_suffix=None, masks_output_dir=None):
    """Crop the foreground region based on mask

    Args:
        images_dir (string): images folder.
        image_suffix_list (list of string): list of suffix of name of image files.
        images_output_dir (string): folder that you want to save cropped images and cropped masks in.
        masks_dir (string): masks folder.
        masks_output_dir (string, optional): folder that you want to save cropped masks in. Defaults to None.
        mask_suffix (string, optional): suffix of name of mask files. Defaults to None.
    """   
    image_suffix_list = ["C0", "DE", "T2"]
    if not os.path.exists(images_output_dir):
        os.makedirs(images_output_dir)
    if masks_output_dir is not None and (not os.path.exists(masks_output_dir)):
        os.makedirs(masks_output_dir)
    margin = [0, 30, 30]
    masks_list = os.listdir(masks_dir)
    masks_list.sort()
    json_dict = OrderedDict()
    for mask in masks_list:
        mask_path = os.path.join(masks_dir, mask)
        if mask.endswith(".nii.gz"):
            print("#" * 11 *11)
            print(mask_path)
            mask_sitk = sitk.ReadImage(mask_path)
            mask_npy = sitk.GetArrayFromImage(mask_sitk)
            mask_shape = mask_npy.shape
            crop_bbox_min, crop_bbox_max = get_ND_bounding_box(mask_npy, margin=margin)
            # do not crop along depth dimension
            crop_bbox_min[0] = 0
            crop_bbox_max[0] = mask_shape[0]
            print(crop_bbox_min, crop_bbox_max)
            json_dict[mask_path] = {"crop_bbox_min": crop_bbox_min, "crop_bbox_max": crop_bbox_max}
            mask_output_npy = crop_ND_volume_with_bounding_box(mask_npy, crop_bbox_min, crop_bbox_max)
            if mask_suffix is not None:
                mask = mask.replace("_" + mask_suffix + ".nii.gz", ".nii.gz")
            if masks_output_dir is not None:
                save_array_as_nifty_volume(mask_output_npy, os.path.join(masks_output_dir, mask), mask_path)
            save_array_as_nifty_volume(convert_label(mask_output_npy, [1, 2, 3, 4, 5], [1, 2, 3, 1, 1]), \
                os.path.join(images_output_dir, mask.replace(".nii.gz", "_{0:04d}.nii.gz".format(len( \
                    image_suffix_list)))), mask_path)
            for i, image_suffix in enumerate(image_suffix_list):
                image = mask.replace(".nii.gz", "_{}.nii.gz".format(image_suffix))
                image_path = os.path.join(images_dir, image)
                print(image_path)
                image_sitk = sitk.ReadImage(image_path)
                image_npy = sitk.GetArrayFromImage(image_sitk)
                image_output_npy = crop_ND_volume_with_bounding_box(image_npy, crop_bbox_min, crop_bbox_max)
                save_array_as_nifty_volume(image_output_npy, os.path.join(images_output_dir, mask.replace( \
                    ".nii.gz", "_{0:04d}.nii.gz".format(i))), image_path)
    save_json(json_dict, os.path.join(images_output_dir, "crop_information.json"))
    if masks_output_dir is not None:
        save_json(json_dict, os.path.join(masks_output_dir, "crop_information.json"))

if __name__ == "__main__":
    if(len(sys.argv) < 2):
        print('Number of arguments should be 2. e.g.')
        print('    python crop_for_fine_stage.py train')
        exit()
    stage = str(sys.argv[1])
    # root_dir = "/home/guotai/disk2t/data/heart/MyoPS/data_preprocessed"
    # nnUNet_data_dir = "/home/guotai/disk2t/projects/nnUNet/exp_dir/nnUNet_raw/nnUNet_raw_data/Task112_MyoPS"
    root_dir = path_dict['MyoPS_data_dir'] + "/data_preprocessed"
    nnUNet_data_dir = path_dict['nnunet_raw_data_dir'] + "/Task112_MyoPS"
    # Task112_MyoPS: use labelsTr(GTs) to crop and use predsTr(predictions) as coarse segmentation
    predsTr_dir = "result/unet2d_post"
    predsTs_dir = "result/unet2d_test_post"
    if stage == "train":
        imagesTr_dir = os.path.join(root_dir, "imagesTr")
        labelsTr_dir = os.path.join(root_dir, "labelsTr")
        labelsTr_suffix = "gd"
        imagesTr_output_dir = os.path.join(nnUNet_data_dir, "imagesTr")
        labelsTr_output_dir = os.path.join(nnUNet_data_dir, "labelsTr")
        crop_acc_mask(imagesTr_dir, imagesTr_output_dir, labelsTr_dir, labelsTr_suffix, labelsTr_output_dir)
        # In imagesTr_output_dir there are "XXXX_0003.nii.gz" derived from labelsTr(GTs) which should be replaced
        # with predsTr(predictions) in predsTr_dir
        json_dict = load_json(os.path.join(imagesTr_output_dir, "crop_information.json"))
        for label_path in json_dict.keys():
            crop_bbox_min = json_dict[label_path]["crop_bbox_min"]
            crop_bbox_max = json_dict[label_path]["crop_bbox_max"]
            label = label_path.split("/")[-1]
            pred = label.replace("_{0:}".format(labelsTr_suffix), "")
            image = label.replace("_{0:}".format(labelsTr_suffix), "_0003")
            pred_path = find(pred, predsTr_dir)
            print(pred, pred_path)
            pred_sitk = sitk.ReadImage(pred_path)
            pred_npy = sitk.GetArrayFromImage(pred_sitk)
            pred_output_npy = crop_ND_volume_with_bounding_box(pred_npy, crop_bbox_min, crop_bbox_max)
            save_array_as_nifty_volume(pred_output_npy, os.path.join(imagesTr_output_dir, image), pred_path)
    elif stage == "test":
        imagesTs_dir = os.path.join(root_dir, "imagesTs")
        imagesTs_output_dir = os.path.join(nnUNet_data_dir, "imagesTs")
        crop_acc_mask(imagesTs_dir, imagesTs_output_dir, predsTs_dir, mask_suffix=None, masks_output_dir=None)
    else:
        raise ValueError("stage can only be train or test")