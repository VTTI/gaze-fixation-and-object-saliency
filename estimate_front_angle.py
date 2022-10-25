import argparse
import numpy as np
import cv2
import time
import yaml
import torch
import torch.nn as nn
from torch.autograd import Variable
import pandas as pd
from torchvision import transforms
from utils import select_device
import torch.backends.cudnn as cudnn
import torchvision
from PIL import Image
from face_detection import RetinaFace
from model import L2CS
import seaborn as sns
import os.path as osp
import matplotlib.pyplot as plt
import gc
import sys

def parse_args():
    """Parse input arguments."""
    parser = argparse.ArgumentParser(
        description='driver gaze fixation')
    parser.add_argument(
        '--file',dest='filename', help='csv file with gaze angles', type=str)
    parser.add_argument(
        '--config',dest='config', help='Path of config file.', 
        default='./config.yaml', type=str)
    parser.add_argument(
            '--output',dest='output', help='Path of output folder.', type=str)

    args = parser.parse_args()
    return args

def eulerToDegrees(euler):
    pi = 22.0/7.0
    return ( (euler) / (2 * pi) ) * 360

                        
    
if __name__ == '__main__':

    args = parse_args()
    fname = "l2cs_"+osp.splitext(osp.basename(args.filename))[0]
    filename=osp.join(args.output, fname + "_gaze.csv")
    
    
    #get config parameters
    with open(args.config) as stream:
        config = yaml.safe_load(stream)
    
    old_yaw=config['front_yaw']
    old_pitch=config['front_pitch']
    config['old_front_yaw']=old_yaw
    config['old_front_pitch']=old_pitch
    df_angles = pd.read_csv(filename)
    #nppitch = np.asarray(pitch, dtype=np.float32)
    pitchax= sns.histplot(df_angles['pitch'], kde=True)
    pitchx = pitchax.lines[0].get_xdata() # Get the x data of the distribution
    pitchy = pitchax.lines[0].get_ydata() # Get the y data of the distribution
    maxid_pitch = np.argmax(pitchy)
    config['front_pitch']=float(pitchx[maxid_pitch])
    plt.clf()
    #plt.savefig("./full_face_results/pitch_"+filename+".png")
    #npyaw = np.asarray(yaw, dtype=np.float32)
    yawax = sns.histplot(df_angles['yaw'], kde=True)
    yawx = yawax.lines[0].get_xdata() # Get the x data of the distribution
    yawy = yawax.lines[0].get_ydata() # Get the y data of the distribution
    maxid_yaw = np.argmax(yawy)
    config['front_yaw']=float(yawx[maxid_yaw])
    print("pitch_front:"+str(pitchx[maxid_pitch]))
    print("yaw_front:"+str(yawx[maxid_yaw]))  
    plt.clf()
    pitch_degrees= sns.histplot(df_angles['pitch'],color="blue",label="pitch",kde=True,alpha=0.5)
    yaw_degrees= sns.histplot(df_angles['yaw'],color="red",label="yaw",kde=True)
    plt.legend()
    plt.savefig(osp.join(args.output,"plot_"+osp.basename(filename).split(".")[0]+".png"))
    print("front angles updated")
    with open(args.config, "w") as f:
        yaml.dump(config, f)
    
    