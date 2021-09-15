"""Script for writing cvs files
"""

import os
import csv
import pandas as pd
import random
from random import shuffle
from path_config import path_dict

def create_csv_file(data_root, output_file, modalities, stage):
    """
    create a csv file to store the paths of files for each patient
    """
    if(stage == "train"):
        img_folder = "imagesTr"
        lab_folder = "labelsTr"
    elif(stage == "test"):
        img_folder = "imagesTs"
        lab_folder = "labelsTs"
    img_dir = os.path.join(data_root, img_folder)
    lab_dir = os.path.join(data_root, lab_folder)
    image_names = os.listdir(img_dir)
    image_names = [item for item in image_names if modalities[0] in item]
    image_names.sort()
    print('total number of images {0:}'.format(len(image_names)))
    img_lab_pairs = []
    for image_name in image_names:
        imgs = [image_name.replace(modalities[0], item) for item in modalities]
        img_paths = [os.path.join(img_folder, item) for item in imgs]
        lab_path  = os.path.join(lab_folder, image_name.replace(modalities[0], 'gd'))
        if(stage == "test" and not os.path.isdir(lab_dir)):
            lab_path = ''
        img_lab_pairs.append(img_paths + [lab_path])

    with open(output_file, mode='w') as csv_file:
        csv_writer = csv.writer(csv_file, delimiter=',', 
                            quotechar='"',quoting=csv.QUOTE_MINIMAL)
        csv_writer.writerow(modalities + ["label"])
        for item in img_lab_pairs:
            csv_writer.writerow(item)

def n_fold_split(csv_file, fold_num):
    output_dir = '/'.join(csv_file.split('/')[:-1])
    random.seed(2019)
    with open(csv_file, 'r') as f:
        lines = f.readlines()
    data_lines = lines[1:]
    shuffle(data_lines)
    N = len(data_lines)
    N_f = N // fold_num
    for fold in range(fold_num):
        if fold == fold_num - 1:
            valid_lines = data_lines[fold * N_f:]
        else:
            valid_lines = data_lines[fold * N_f : (fold + 1) * N_f]
        train_lines = [item for item in data_lines if item not in valid_lines]
        
        with open(output_dir + "/fold{0:}_train.csv".format(fold + 1), 'w') as f:
            f.writelines(lines[:1] + train_lines)
        with open(output_dir + "/fold{0:}_valid.csv".format(fold + 1), 'w') as f:
            f.writelines(lines[:1] + valid_lines)

def get_evaluation_image_pairs(test_csv, gt_seg_csv):
    with open(test_csv, 'r') as f:
        input_lines = f.readlines()[1:]
        output_lines = []
        for item in input_lines:
            gt_name = item.split(',')[-1]
            gt_name = gt_name.rstrip()
            seg_name = gt_name.split('/')[-1]
            seg_name = seg_name.replace("_gd.nii.gz", ".nii.gz")
            output_lines.append([gt_name, seg_name])
    with open(gt_seg_csv, mode='w') as csv_file:
        csv_writer = csv.writer(csv_file, delimiter=',', 
                            quotechar='"',quoting=csv.QUOTE_MINIMAL)
        csv_writer.writerow(["ground_truth", "segmentation"])
        for item in output_lines:
            csv_writer.writerow(item)

if __name__ == "__main__":
    # create cvs file for training and testing images
    data_dir  = path_dict["MyoPS_data_dir"]  + "/data_preprocessed"
    modalities = ["C0", "DE", "T2"]
    for stage in ["train", "test"]:
        output_file = "config/data/data_{0:}.csv".format(stage)
        create_csv_file(data_dir, output_file, modalities, stage)
        if(stage == "train"):
            n_fold_split(output_file, 5)

    # for evaluation of the cross validation results only
    get_evaluation_image_pairs("config/data/data_train.csv","config/data/eval_gt_seg.csv")

