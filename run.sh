#!/bin/bash

python3 get_gaze.py --face "$1" --front "$2" --gpu $3 --config "$4" --output "$5"
python3 estimate_front_angle.py --file "$1" --output "$5"
#change the cutoff, nyq freq and order for filter_csv file here
python3 filter_csv.py "$1" "$5" "2" "15" "4"
python3 gaze_overlay.py --face "$1" --front "$2" --output_folder "$5" --config "$4" --obj_det --pan_seg

