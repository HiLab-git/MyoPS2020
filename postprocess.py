import numpy as np
import os
import sys
import SimpleITK as sitk
from scipy import ndimage
from pymic.util.image_process import get_largest_component, convert_label
from pymic.io.image_read_write import save_array_as_nifty_volume

def postprocess(origin_folder, output_folder):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    preds_list = os.listdir(origin_folder)
    for pred in preds_list:
        pred_path = os.path.join(origin_folder, pred)
        if pred.endswith(".nii.gz"):
            print(pred_path)
            pred_sitk = sitk.ReadImage(pred_path)
            pred_npy = sitk.GetArrayFromImage(pred_sitk)
            pred_binary = (pred_npy != 0)
            # get_largest_component for all_classes
            pred_binary_post = get_largest_component(pred_binary)
            pred_npy_post_all_classes = pred_npy * pred_binary_post
            # get_largest_componet for per_class(1, 2, 3)
            pred_npy_post_per_class = np.zeros_like(pred_npy_post_all_classes)
            for cls in range(1, 4):
                pred_binary = (pred_npy_post_all_classes == cls)
                pred_binary_post = get_largest_component(pred_binary)
                pred_npy_post_per_class += pred_npy_post_all_classes * pred_binary_post
            save_array_as_nifty_volume(pred_npy_post_per_class, os.path.join(output_folder, pred), pred_path)

if __name__ == "__main__":
    if(len(sys.argv) < 3):
        print('Number of arguments should be 2. e.g.')
        print('    python postprrcess.py input_folder  output_folder')
        exit()
    input_folder, output_folder = str(sys.argv[1]), str(sys.argv[2])
    if(not os.path.isdir(output_folder)):
        os.mkdir(output_folder)
    postprocess(input_folder, output_folder)