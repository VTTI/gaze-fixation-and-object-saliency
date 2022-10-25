# HOW TO USE Driver Gaze Fixation

# STEP 1

● Clone the github directory
``` 
Git clone “https://gitlab.vtti.vt.edu/ctbs/fhwa-cv/drivergazefixation.git”
```
● Build the docker file:
```
cd drivergazefixation
docker build -t drivergaze .
```
● Run the docker image:
```
docker run -it --rm --runtime=nvidia -v /vtti:/vtti --user 12731:
drivergaze:latest /bin/bash

NOTE: runtime arg is important. Keep user ID that has access to input and output folder. Make sure to -v the directory that is the origin for all the paths you mention or refactor paths accordingly.

```
● After running docker, CD to the path where you cloned drivergazefixation

● For example: cd /vtti/scratch/hbhagat/drivergazefixation

# STEP 2

First of all, we need to set some parameters in the config.yaml

● Edit the config file to specify which object detection model to use

    ○ det_model_checkpoint is the path to the .pth file or the model file

    ○ det_model_config is the path to the config file for the specific model

    ○ det_model_classes refers to the txt file with classes associated with the model

    ○ det_model_threshold is the cutoff threshold for the detection results

    ○ det_gpu_id is the gpu that the detection model will use

    ○ front_pitch is the estimated forward view pitch value (use estimate_front_angle.py if angles are not known)

    ○ front_yaw is the estimated forward view yaw value (use estimate_front_angle.py if angles are not known)

    ○ dist is the estimated distance value in pixels from driver's face to the camera calculated using the given formula (refer to the pdf for indepth information)

    ○ snapshot_path is the path to L2CS-Net gaze model

    ○ arch is the architecture used to train the model

    ○ batch_size is batch size of the model

    ○ f_picth is adjustments to the forward pitch

    ○ f_yaw is the adjustments to the forward yaw
    

# STEP 3

There are 3 files that are run sequentially using ./run.sh. The get_gaze.py file is used to record the gaze angles and the face coordinates in a csv file with l2cs_* as a prefix. Next filter_csv.py file uses the csv file generated previously, and smooths out the jittering in gaze angles using a order 5 butterworth filter with cutoff as 2 (could be changed in function in the file). And lastly, heatmap_overlay.py file calculates the pix_x and pix_y for the gaze points from the filtered csv and generates a heatmap to overlay on the forward view. The outcome is a merged video with gaze point and heatmap.

./run.sh takes 5 arguments

○ $1 is the path to the face view video file

○ $2 is the path to the forward view video file

○ $3 is the gpu id

○ $4 is confog.yaml file path

○ $5 is the path of the output directory (to store result)

Example Command: 
./run.sh "/vtti/projects03/451600/Working/Task_2_3/Mask_Validation/nonclippedMaskValidation/full_face_video/fullFaceViewVideoV1/CMVP_0000_0000_10_130218_1429_00079_Face.mp4" "/vtti/projects03/451600/Working/Task_2_3/Mask_Validation/nonclippedMaskValidation/SHRP2 video/SHRP2FrontViewVideoV1/CHPV_0000_0000_10_130218_1924_00088_Front.mp4" "0" "./config.yaml" "/vtti/scratch/hbhagat/driver_gaze_fixation/"


        

