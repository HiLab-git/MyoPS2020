# Winner of MyoPS 2020 Challenge
[PyMIC_link]:https://github.com/HiLab-git/PyMIC
[nnUNet_link]:https://github.com/MIC-DKFZ/nnUNet
This repository provides source code for myocardial pathology segmentation (MyoPS) Challenge 2020. The method is detailed in the [paper](https://link.springer.com/chapter/10.1007/978-3-030-65651-5_5), and it won the 1st place of [MyoPS 2020](http://www.sdspeople.fudan.edu.cn/zhuangxiahai/0/myops20). Our code is based on [PyMIC][PyMIC_link], a pytorch-based toolkit for medical image computing with deep learning, that is lightweight and easy to use, and [nnUNet][nnUNet_link], a self-adaptive segmentation method for medical images.

<img src='./picture/method.png'  width="400">

## Requirements
Some important required packages include(our test environment):
* Python >= 3.6.9
* [Pytorch](https://pytorch.org) >=1.7.1
* [PyMIC][PyMIC_link] >= 0.2.4
* Some basic python packages such as Numpy, Scikit-image, SimpleITK, Scipy ......
* Download [nnUNet][nnUNet_link], and put them in the `ProjectDir` such as `/mnt/data1/swzhai/projects/MyoPS2020`.

## Configure data directories and environmental variables
* Configure data directories in `path_confg.py`:
``` bash
import os 
path_dict = {}
path_dict['MyoPS_data_dir'] = "/mnt/data1/swzhai/dataset/MyoPS/"
path_dict['nnunet_raw_data_dir'] = "/mnt/data1/swzhai/dataset/MyoPS/nnUNet_raw_data_base/nnUNet_raw_data"
```
where `raw_data_dir` is the path of raw data for the MyoPS dataset, and `nnunet_raw_data_dir` is the path of raw data used by nnU-Net in the second stage of our method.
* Install [nnUNet][nnUNet_link] and set environment variables.
```bash
cd nnUNet
pip install -e .
export nnUNet_raw_data_base="DataDir/nnUNet_raw_data_base"
export nnUNet_preprocessed="DataDir/nnUNet_preprocessed"
export RESULTS_FOLDER="ProjectDir/result/nnunet"

# in my case
export nnUNet_raw_data_base="/mnt/data1/swzhai/dataset/MyoPS/nnUNet_raw_data_base"
export nnUNet_preprocessed="/mnt/data1/swzhai/dataset/MyoPS/nnUNet_preprocessed"
export RESULTS_FOLDER="/mnt/data1/swzhai/projects/MyoPS/myops/result_nnunet"
```

## Dataset and Preprocessing
Download the dataset from [MyoPS 2020](http://www.sdspeople.fudan.edu.cn/zhuangxiahai/0/myops20) and put the dataset in the path_dict['MyoPS_data_dir'] such as `/mnt/data1/swzhai/dataset/MyoPS`, specifically, `MyoPS_data_dir/data_raw/imagesTr` for training images, `MyoPS_data_dir/data_raw/labelsTr` for training ground truth and `MyoPS_data_dir/data_raw/imagesTs` for test images.

For data preprocessing, Run:
```bash
python crop_for_coarse_stage.py
```
This will crop the images with the maximal bounding box in the training set, and the cropped results are saved in `DataDir/data_preprocessed/imagesTr`, `DataDir/data_preprocessed/labelsTr` and `DataDir/data_preprocessed/imagesTs` respectively. `crop_information.json` in each folder contains bounding box coordinates that will be used at the fine segmentation stage. 

## Coarse segmentation Model
The segmentation algorithm uses a coarse-to-fine method. [PyMIC][PyMIC_link] and [nnUNet][nnUNet_link] are used in the coarse and fine stages, respectively. In the coarse segmentation stage, we use 2D U-Net to segment four classes: background, complete ring-shaped myocardium, left ventricular blood pool and rigth ventricular blood pool. The network is trained with a combination of Dice loss and  cross entropy loss.

### Training and cross validation
* We use five-fold cross validation for training and validation of the coarse model. Run the following command to create csv files of training and testing datasets that are required by PyMIC, and split the training data into five folds. You need to reset the value of `root_dir` based on your machine.
```bash
python write_csv_files.py
```
The csv files will be saved to `config/data`.
* For training and validation of the first fold, run:
```bash
python myops_run.py train config/train_val.cfg 1
python myops_run.py test  config/train_val.cfg 1
```
The segmentation model will be saved in `model/unet2d/fold_1`, and prediction of the validation data for the first fold will be saved in `result/unet2d`.
* Similarly to the above step, run the training and inference code for fold_2-fold_5.
* After training and inference for all the five folds, to see the performance of the five fold cross validation, set `ground_truth_folder_root` to the correct value in `config/evaluation.cfg`, and run:
```bash
pymic_evaluate_seg config/evaluation.cfg
```
We can see the Dice of class 1, 2 and 3, respectively.
* For post processing, run:
```bash
python  postprocess.py result/unet2d result/unet2d_post
```
The results will be saved in `result/unet2d_post`. You can set `segmentation_folder_root  = result/unet2d_post` in `config/evaluation.cfg` and run the evaluation code again. The average dice scores before after post processing on my machine are:
|---|class_1|class_2|class_3|average|
|---|---|---|---|---|
|w/o pp|0.8709|0.9050|0.9076|0.8945|
|w/ pp|0.8770|0.9117|0.9128|0.9005|

### Inference for testing data
* We use an ensemble of five models obtained during the five-fold cross validation for inference. Open `config/test.cfg` and set `ckpt_name` to the list of the best performing checkpoints of the five folds. The best performing iteration number for fold i can be found in `model/unet2d/fold_i/model_best.txt`. Run the following command for inference:
```bash
python myops_test.py test config/test.cfg
```
The results will be saved in `result/unet2d_test`.
* Run this command to post process the segmentation of the testing images:
```bash
python postprocess.py result/unet2d_test result/unet2d_test_post
```

## Fine segmentation
In the fine segmentation stage, we use nn-UNet to segment all the classes. This section is highly dependent on [nnUNet][nnUNet_link], so make sure that you have some basic experience of using [nnUNet][nnUNet_link] before you do the following operations.

Tips: In order to save unnecessary time, you can change `self.max_num_epochs = 1000` to `self.max_num_epochs = 300` in `nnUNet/nnunet/training/network_training/nnUNetTrainerV2.py`.

### Data preparation
* The coarse segmentation will serve as an extra channel for the input of the network, i.e., the first 3 modalities are C0(_0000), DE(_0001) and T2(_0002), respectively. The 4th modality(_0003) is the coarse segmentation result. 

* Run the following commands to prepare training and testing data for nn-UNet:
```
python crop_for_fine_stage.py train
python crop_for_fine_stage.py test
python create_dataset_json.py
```

### training
* Dataset conversion and preprocess. Run:
```bash
nnUNet_plan_and_preprocess -t 112 --verify_dataset_integrity
```
* Train 2D UNet. For FOLD in [0, 1, 2, 3, 4], run:
```bash
nnUNet_train 2d nnUNetTrainerV2 Task112_MyoPS FOLD --npz
```
* Train 2.5D(3D) UNet. For FOLD in [0, 1, 2, 3, 4], run:
```bash
nnUNet_train 3d_fullres nnUNetTrainerV2 Task112_MyoPS FOLD --npz
```
### inference
Here we have 2 fine models(i.e. 2D UNet and 2.5D UNet). Run:
```bash
nnUNet_find_best_configuration -m 2d 3d_fullres -t 112
```
The terminal will output some commands that are used to infer test dataset and get their ensemble. In my case, I get the following commands: 
```bash
nnUNet_predict -i FOLDER_WITH_TEST_CASES -o OUTPUT_FOLDER_MODEL1 -tr nnUNetTrainerV2 -ctr nnUNetTrainerV2CascadeFullRes -m 2d -p nnUNetPlansv2.1 -t Task112_MyoPS

nnUNet_predict -i FOLDER_WITH_TEST_CASES -o OUTPUT_FOLDER_MODEL2 -tr nnUNetTrainerV2 -ctr nnUNetTrainerV2CascadeFullRes -m 3d_fullres -p nnUNetPlansv2.1 -t Task112_MyoPS

nnUNet_ensemble -f OUTPUT_FOLDER_MODEL1 OUTPUT_FOLDER_MODEL2 -o OUTPUT_FOLDER -pp /mnt/data1/swzhai/projects/MyoPS/myops/result_nnunet/nnUNet/ensembles/Task112_MyoPS/ensemble_2d__nnUNetTrainerV2__nnUNetPlansv2.1--3d_fullres__nnUNetTrainerV2__nnUNetPlansv2.1/postprocessing.json

# in my case
nnUNet_predict -i /mnt/data1/swzhai/dataset/MyoPS/nnUNet_raw_data_base/nnUNet_raw_data/Task112_MyoPS/imagesTs -o /mnt/data1/swzhai/projects/MyoPS/myops/result_nnunet/test_2D -tr nnUNetTrainerV2 -ctr nnUNetTrainerV2CascadeFullRes -m 2d -p nnUNetPlansv2.1 -t Task112_MyoPS --save_npz

nnUNet_predict -i /mnt/data1/swzhai/dataset/MyoPS/nnUNet_raw_data_base/nnUNet_raw_data/Task112_MyoPS/imagesTs -o /mnt/data1/swzhai/projects/MyoPS/myops/result_nnunet/test_3D -tr nnUNetTrainerV2 -ctr nnUNetTrainerV2CascadeFullRes -m 3d_fullres -p nnUNetPlansv2.1 -t Task112_MyoPS --save_npz

nnUNet_ensemble -f /mnt/data1/swzhai/projects/MyoPS/myops/result_nnunet/test_2D /mnt/data1/swzhai/projects/MyoPS/myops/result_nnunet/test_3D -o /mnt/data1/swzhai/projects/MyoPS/myops/result_nnunet/test_ensemble -pp /mnt/data1/swzhai/projects/MyoPS/myops/result_nnunet/nnUNet/ensembles/Task112_MyoPS/ensemble_2d__nnUNetTrainerV2__nnUNetPlansv2.1--3d_fullres__nnUNetTrainerV2__nnUNetPlansv2.1/postprocessing.json --npz
```
Replace `FOLDER_WITH_TEST_CASES` with the test dataset folder `DataDir/nnUNet_raw_data_base/nnUNet_raw_data/Task112_MyoPS/imagesTs`, replace `OUTPUT_FOLDER_MODEL1` with 2D model folder `ProjectDir/myops/result_nnunet/test_2D`, replace `OUTPUT_FOLDER_MODEL2` with 3D model folder `ProjectDir/myops/result_nnunet/test_3D`, replace `OUTPUT_FOLDER` with ensemble folder `ProjectDir/myops/result_nnunet/test_ensemble` and run above commands.

Notice: Add arguments "--save_npz" and "--npz" to save .npz file which are model probability for future ensemble.

Because we crop the images twice in the whole process, we need to insert the cropped images into the original images by using `crop_information.json`. Set your foler path and Run:
```bash
python get_final_test.py
```

## Citation
```
@inproceedings{zhai2020myocardial,
  title={Myocardial edema and scar segmentation using a coarse-to-fine framework with weighted ensemble},
  author={Zhai, Shuwei and Gu, Ran and Lei, Wenhui and Wang, Guotai},
  booktitle={Myocardial Pathology Segmentation Combining Multi-Sequence CMR Challenge},
  pages={49--59},
  year={2020},
  organization={Springer}
}
```
***This README is to be improved and questions are welcome.***
