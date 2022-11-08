# Driver Gaze Fixation and Object Saliency
This repository provides code to estimate Point of Gaze (PoG) on the scene-facing (front video) view w.r.t the gaze angle obtained from the driver-faciing (face video) view. We also provide other functionalities listed below. Please note that the code is tested on the specified models but you could change them according to your needs.

1. Closest object of fixation from PoG using Panoptic Segmentation (Detectron2)
2. Closest object of fixation from PoG using Object Detection (SHRP2 trained models)
3. Fixation Heatmap (Heatmap generated for the current frame by aggregating the PoGs from the last (fps x 3) frames)
4. Yarbus Plots

## Installation
To install the framework the recommened method is by following these steps:

Install Docker (https://www.docker.com/).

Create a new file named "Dockerfile" and copy inside the contents of "docker/Dockerfile".

Run: docker build . -t gazefixation

Run: nvidia-docker run -it --gpus=all --rm -v "path_to_local_folder":"path_inside_docker" gazefixation bash

# Setting up the config file
Below is an examaple of the config file.
``` 
# gaze angle estimation model path, architecture and batch size (we use L2CSNet's specifications)
snapshot_path: ./models/Gaze360/L2CSNet_gaze360.pkl
arch: ResNet50
batch_size: 1

# object detection model's specifications (we have used our inhouse SHRP2 object detection model)
det_gpu_id: 1
det_model_checkpoint: /vtti/scratch/mmdetection/trained_models/shrp2_original_split+extra_data_outside_objects/epoch_10.pth
det_model_classes: /vtti/scratch/mmdetection/classes/classes_shrp2_extra.txt
det_model_config: /vtti/scratch/mmdetection/custom_configs/shrp2+extra/cascade_rcnn_r101_fpn_dconv_c3-c5_1x_coco.py
det_model_threshold: 0.5

# dist is the distance in pixels from driver's face to the camera
dist: 480
# f_picth and f_yaw are the correction of front camera view 
f_pitch: -0.04
f_yaw: 0.15
# front_picth and front_yaw are the angles calculated by estimate_front_angle.py file and are automatically updated in the config file when executed
front_pitch: -0.4002903467336685
front_yaw: 0.03331976381909563

# These are just stored to retain previous calculations of front_picth and front_yaw
old_front_pitch: -0.4002903467336685
old_front_yaw: 0.03331976381909563

# Segmentation model configurations are stored here. (we use detectron2's panoptic model)
seg_confidence_threshold: 0.5
seg_config_file: configs/COCO-PanopticSegmentation/panoptic_fpn_R_101_3x.yaml
seg_opts:
- MODEL.WEIGHTS
- ./models/detectron2/panoptic_fpn_R_101_3x/139514519/model_final_cafdb1.pkl
seg_overlay: 1
```
# Demo Example
This command runs the algorithm on the example video files in the Videos folder. It performs both segmentation and detection and stores the result in the current directory.
```
./run.sh ./Videos/face_ex.mp4 ./Videos/front_ex.mp4 0 ./config.yaml .
```
# Usage

You could directly use the run.sh command to get the desired output.
```
./run.sh <face_video_file_path> <front_video_file_path> <gpu_id> <config_file_path> <output_directory>
```
Note: for filtering we use default values for 15 fps
cutoff: 2
nyq frequency: 7.5
order of butterworth filter: 4

If you'd like to run the files seperately:

### get_gaze.py
The L2CSNet's Pretrained Gaze360 model (https://github.com/Ahmednull/L2CS-Net) was used to get the gaze angles. First we obtain the gaze angles and store them in a seperate CSV file. The parameters needed to obtain the gaze angles are the face video path, front video path (just to get the frame width), config file path and output directory path.

```
python3 get_gaze.py --face <face_video_file_path> --front <front_video_file_path> --gpu <gpu_id> --config <config_file_path> --output <output_directory_path>
```
### estimate_front_angle.py
The next step is estimating the front angle. This can be done by executing the estimating_front_angle.py file like so. It needs the gaze angles csv file and an output directory path as arguments.

```
python3 estimate_front_angle.py --file <gaze_angles_csv> --output <output_directory_path>
```

### filter_csv.py
Since gaze angle estimation is done on a frame by frame basis we filter angles using a butterworth filter. filter_csv.py file takes the arguments of gaze angles, output directory, cutoff frequency, nyq frequency and order of the butterworth filter. As an output it saves a csv file with filtered gaze angles.

```
python3 filter_csv.py <gaze_angles_csv> <output_directory_path> <cutoff_frequency> <nyq frequency> <order_of_filter>

```

### gaze_overlay.py
gaze_overlayl.py file is used to stich the face and front video side by side. gaze angles from the csv file are taken and transfored to PoG on the front video frames. This file also takes arguments of --obj_det and --pan_seg. if you specify the arguments, the code will perform object detection and segementation respectively, else it will just perform the PoG transformation and save the video output file. It takes the arguments of the face video file, the front video file, output folder path, config file path and optional arguments : pan_seg, obj_det (give these if you want the funcationality in the output)

```
python3 gaze_overlay.py --face <face_video_file_path> --front <front_video_file_path> --config <config_file_path> --output_folder <output_directory_path> --obj_det --pan_seg

```
### heatmap_overlay.py
The output is generared with heatmaps over the PoG estimates.

```
python3 heatmap_overlay.py --face <face_video_file_path> --front <front_video_file_path> --config <config_file_path> --output <output_directory_path>
```


# Demo results
## output with panoptic segmentation
![](https://github.com/VTTI/gaze-fixation-and-object-saliency/blob/master/gaze_fixation_example.gif)

# Yarbus-style visualization

## Required Files: Saliency

## 1.) CSV
	Adjusted to Windshield Camera CSV files (detections and gaze location):
	
	e.g., dgf_l2cs_CMVP_0000_0000_10_130218_1429_00079_Face_adjusted_bbox.csv
	
	Frame#	X-value for Gaze	Y-value for Gaze	Object Name

## 2.) Frames for the video
	Directory containing zero-preceeding frame numbers e.g., 000000- 999999

## 3.) Output dir

# Visualizations (Front cam)

![](https://github.com/VTTI/gaze-fixation-and-object-saliency/blob/master/Videos/Signalized_left_turn_yarbus.gif)
