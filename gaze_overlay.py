import csv
import cv2
import yaml
import numpy as np
import math as m
import time
import calibration
from face_detection import RetinaFace
from mmdet.apis import init_detector, inference_detector
import sys
from utils import select_device, draw_gaze
import os.path as osp
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from scipy.ndimage.filters import gaussian_filter
import os
import cv2
import time
import tqdm
import json
import glob
import tempfile
import warnings
import argparse
import mimetypes
import numpy as np
import multiprocessing as mp

from detectron2.config import get_cfg
from detectron2.data.detection_utils import read_image
from detectron2.utils.logger import setup_logger
from detectron2.utils.visualizer import ColorMode

from predictor2 import VisualizationDemo


# constants
WINDOW_NAME = "COCO detections"

def parse_args():
    """Parse input arguments."""
    parser = argparse.ArgumentParser(
        description='driver gaze fixation')
    parser.add_argument(
        '--face',dest='face', type=str)
    parser.add_argument(
        '--front',dest='front', type=str)
    parser.add_argument(
        '--output_folder',dest='output_folder', type=str)
    parser.add_argument(
        '--config',dest='config', type=str)
    parser.add_argument(
        '--trial_id',dest='trial_id', type=int, default=0)
    parser.add_argument(
        '--obj_det',dest='obj_det', action="store_true")
    parser.add_argument(
        '--pan_seg',dest='pan_seg', action="store_true")

    args = parser.parse_args()
    return args
def setup_cfg(config):
    # load config from file and command-line arguments
    cfg = get_cfg()

    # To use demo for Panoptic-DeepLab, please uncomment the following two lines.
    # from detectron2.projects.panoptic_deeplab import add_panoptic_deeplab_config  # noqa
    # add_panoptic_deeplab_config(cfg)

    cfg.merge_from_file(config['seg_config_file'])
    cfg.merge_from_list(list(config['seg_opts']))

    # Set score_threshold for builtin models
    cfg.MODEL.RETINANET.SCORE_THRESH_TEST = config['seg_confidence_threshold']
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = config['seg_confidence_threshold']
    cfg.MODEL.PANOPTIC_FPN.COMBINE.INSTANCES_CONFIDENCE_THRESH = config['seg_confidence_threshold']
    cfg.freeze()

    return cfg


def drawfront(frame_front,front_pitch,front_yaw,pix_x,pix_y,pitch_predicted,yaw_predicted,dist):
    
    R = calibration.Rx(front_pitch)*calibration.Ry(front_yaw)
    point = np.array([[pix_x],[pix_y],[dist]])
    current_R = calibration.Rx(pitch_predicted)*calibration.Ry(yaw_predicted)
    rotated_R=np.matmul(current_R,R.transpose())
    row_add = np.array([0, 0, 0,1])
    col_add= np.array([0,0,0])
    mapped_point=rotated_R * point
    
    #print(mapped_point)
    return int(mapped_point[0,0]),int(mapped_point[1,0]),frame_front

def closest_object(result,xi,yi,classes,det_threshold):
    distances={}
    for i,v in enumerate(result):
                #print("class:"+classes[i])
                for j,y in enumerate(v):
                    if(round(y[4],2)>=det_threshold):
                        xmindif=int(y[0])-xi
                        xmaxdif=xi-int(y[2])
                        ymindif=int(y[1]) - yi
                        ymaxdif=yi - int(y[3])
                        dx=max(xmindif, 0, xmaxdif)
                        #dx=np.max(x)
                        dy=max(ymindif, 0, ymaxdif)
                        #dy = np.max(y)
                        distances[m.sqrt(dx*dx+dy*dy)]=[classes[i],y[0],y[1],y[2],y[3]]
    if(len(distances)>0):
        #print(min(distances))
        return distances[min(distances)]
    else:
        return
    
def myplot(x, y, s,dim):
    heatmap, xedges, yedges = np.histogram2d(x, y, bins=dim)
    heatmap = gaussian_filter(heatmap, sigma=s)

    #extent = [0, 480, 0, 360]
    return heatmap.T


args = parse_args()
face=args.face
front=args.front
output_folder=args.output_folder
trial_id=args.trial_id


face_cap = cv2.VideoCapture(face)
front_cap = cv2.VideoCapture(front)

fps=14

filename_full = "dgf_"+osp.basename(face)
outputFile = osp.join(output_folder,filename_full)

frame_w=int(front_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_h=int(front_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))


csv_file=osp.join(output_folder,"filtered_l2cs_"+osp.splitext(osp.basename(face))[0] + "_gaze.csv")

fourcc = cv2.VideoWriter_fourcc('M','J','P','G')
vid_writer = cv2.VideoWriter(outputFile, 0x7634706d, 15, (int(front_cap.get(cv2.CAP_PROP_FRAME_WIDTH))*2, int(front_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))))

output = np.zeros((frame_h, frame_w * 2, 3), dtype="uint8")


with open(args.config) as stream:
    config = yaml.safe_load(stream)

os.environ["CUDA_VISIBLE_DEVICES"] = str(config['gpu'])

f_pitch=config['f_pitch']
f_yaw=config['f_yaw']

dist=config['dist']

front_yaw=config['front_yaw'] - f_yaw
front_pitch=config['front_pitch'] - f_pitch

if(args.obj_det):
    det_model_config=config['det_model_config']
    det_model_classes=config['det_model_classes']
    det_model_checkpoint=config['det_model_checkpoint']
    det_model_threshold=config['det_model_threshold']
    det_model_id=config['det_gpu_id']
    det_model = init_detector(det_model_config, det_model_checkpoint, device='cuda:'+str(0))

if(args.pan_seg):
    seg_overlay=config['seg_overlay']
    setup_logger(name="fvcore")
    cfg = setup_cfg(config)

    if seg_overlay == 0:
        seg_overlay = False
    else:
        seg_overlay = True

    demo = VisualizationDemo(cfg, overlay=seg_overlay)

with open(det_model_classes, 'r') as f:
    classes = tuple(f.read().splitlines())
    
with open(csv_file) as fd:
    reader = fd.readlines()
    
obj_bbox = open(osp.join(output_folder,"obj_data"+osp.splitext(osp.basename(face))[0])  + ".csv", "w+")

obj_bbox.write("trail_id,frame_no,obj_class\n")

i=1
j=0
slide_x=[]
slide_y=[]
csvout_bbox_real = open(osp.join(output_folder,"dgf_"+osp.splitext(osp.basename(face))[0])  + "_real.csv", "w+")
csvout_bbox_adjusted = open( osp.join(output_folder,"dgf_"+osp.splitext(osp.basename(face))[0]) + "_adjusted.csv", "w+")
print("overlay in progress..")
with open(det_model_classes, 'r') as f:
    classes = tuple(f.read().splitlines())
record_heatmap=fps*3
while face_cap.isOpened():
            ret, frame = face_cap.read()
            r,frame_front=front_cap.read()
            #cv2.imwrite("./img.jpg",frame)
            if ret==True:
                start_fps = time.time()
                #frame = np.rot90(frame,1)
                #print(frame_w)
                frame = cv2.resize(frame, (frame_w,frame_h))
                row=reader[i].split(',')
                next_row=reader[i+1].split(',')
                frame_no=int(float(row[0]))
                next_frame_no=int(float(next_row[0]))
                while(True):
                    if(int(float(reader[i].split(',')[0]))==int(float(reader[i+1].split(',')[0]))):
                        #print(i)
                        i=i+1
                    else:
                       break
                #print(j)
                #print(frame_no)
                if(j==frame_no):
                    if(args.obj_det):
                        result = inference_detector(det_model, frame_front)
                    i=i+1
                    pitch=float(row[1])
                    yaw=float(row[2])
                    fx_min=float(row[3])
                    fy_min=float(row[4])
                    fx_max=float(row[5])
                    fy_max=float(row[6])
                    bbox_width = fx_max - fx_min
                    bbox_height = fy_max - fy_min
                    fx=(frame_w)//2
                    fy=(frame_h)//2-60
                    
                    p=m.degrees(pitch)
                    y=m.degrees(yaw)
                    pix_x,pix_y,front=drawfront(frame_front, front_pitch, front_yaw, fx, fy, pitch, yaw, dist)
                    #csvout_bbox_real.write('%d,%d,%d,%s' % (frame_no,pix_x,pix_y,) + "\n")
                    draw_gaze(fx_min,fy_min,bbox_width, bbox_height,frame,(yaw,pitch),color=(0,0,255))
                    if(pix_y>=frame_h-60 or pix_x>frame_w):
                        cv2.putText(front, 'Gaze out of frame', (10, 20),cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 255, 0), 1, cv2.LINE_AA)
                        
                        output[0:frame_h, frame_w:frame_w*2]=frame
                        output[0:frame_h, 0:frame_w]=frame_front
                        vid_writer.write(output)
                    else:
                        cv2.putText(frame, 'Pitch: {:.1f}, Yaw:{:.1f}'.format(p,-y), (10, 20),cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 255, 0), 1, cv2.LINE_AA)
                        
                        
                        
                        if(pix_y>=frame_h-60 or pix_x<=0 or pix_y<=0 or pix_x>=frame_w):
                            csvout_bbox_real.write('%d,%d,%d,%s' % (frame_no,pix_x,pix_y,"OOF") + "\n")
                            cv2.putText(front, 'Gaze out of frame', (10, 20),cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 255, 0), 1, cv2.LINE_AA)
                            
                        else:
                            if(args.obj_det):
                                closest=closest_object(result, pix_x, pix_y, classes, det_model_threshold)
                                if(closest):
                                    objclass,xmin,ymin,xmax,ymax=closest
                                    obj_bbox.write(str(args.trial_id)+","+str(frame_no)+","+objclass+"\n")
                                    front=cv2.rectangle(front, (int(xmin),int(ymin)), (int(xmax),int(ymax)),(0, 255, 0), 2)
                                    front = cv2.putText(front, objclass, (int(xmin), int(ymin)-5), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 255, 0), 1)
                            
                            if(args.pan_seg):
                                predictions, visualized_output,new_index,label = demo.run_on_image(front,pix_x,pix_y)
                                
                                #print(new_index)
                                
                                if(label=="N/A"):
                                    #pix_y=pix_y+300
                                    csvout_bbox_real.write('%d,%d,%d,%s' % (frame_no,pix_x,pix_y,label) + "\n")
                                    #cv2.putText(front, 'Gaze out of frame', (10, 20),cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 255, 0), 1, cv2.LINE_AA)
                                    #result.write('%s,%s' % (frame_no,'OOF') + "\n")
                                    front = cv2.circle(front,(pix_x,pix_y), radius=4, color=(0, 0, 255), thickness=-1)
                                    csvout_bbox_adjusted.write('%d,%d,%d,%s,%s' % (frame_no,pix_x,pix_y,label,str(args.trial_id) + "\n"))
        
                                else:
                                    front=visualized_output.get_image()
                                    #(*args, **kwargs):
                                    csvout_bbox_real.write('%d,%d,%d,%s' % (frame_no,pix_x,pix_y,label) + "\n")
                                    csvout_bbox_adjusted.write('%d,%d,%d,%s,%s' % (frame_no,new_index[0],new_index[1],label,str(args.trial_id) + "\n"))
                                    front = cv2.circle(front,new_index, radius=4, color=(0, 0, 255), thickness=-1)
                                    cv2.putText(front,label, (100, 40),cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 255, 0), 1, cv2.LINE_AA)
                        if(not args.pan_seg):
                            front = cv2.circle(front,(pix_x,pix_y), radius=4, color=(0, 0, 255), thickness=-1)
                            csvout_bbox_real.write('%d,%d,%d,%s' % (frame_no,pix_x,pix_y,label) + "\n")
                            csvout_bbox_adjusted.write('%d,%d,%d,%s,%s' % (frame_no,pix_x,pix_y,label,str(args.trial_id) + "\n"))

                            
                                    
                        
                        #cv2.putText(frame, 'Converted Pitch: {:.1f}, Yaw:{:.1f}'.format(m.degrees(),-), (10, 20),cv2.FONT_HERSHEY_COMPLEX_SMALL, 2, (0, 255, 0), 2, cv2.LINE_AA)
                        output[0:frame_h, frame_w:frame_w*2]=frame
                        output[0:frame_h, 0:frame_w]=front
                        vid_writer.write(output)
                else:
                    csvout_bbox_real.write('%d,%d,%d,%s' % (frame_no,pix_x,pix_y,"No gaze angle") + "\n")
                    csvout_bbox_adjusted.write('%d,%d,%d,%s,%s' % (frame_no,pix_x,pix_y,"no gaze angle",str(args.trial_id) + "\n"))

                    output[0:frame_h, frame_w:frame_w*2]=frame
                    output[0:frame_h, 0:frame_w]=frame_front
                    vid_writer.write(output)
                j=j+1

                
            else:
                break
print("done")            
obj_bbox.close()            
face_cap.release()
front_cap.release()
cv2.destroyAllWindows()
