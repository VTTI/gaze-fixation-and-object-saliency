import argparse
import numpy as np
import cv2
import time
import yaml
import torch
import torch.nn as nn
from torch.autograd import Variable
import math as m
from torchvision import transforms
import torch.backends.cudnn as cudnn
import torchvision
import calibration
from PIL import Image
from utils import select_device, draw_gaze
from PIL import Image, ImageOps
from mmdet.apis import init_detector, inference_detector
from annotate import annotate_image
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
        '--gpu',dest='gpu', help='GPU device id to use [0]',
        default="0", type=str)
    parser.add_argument(
        '--face',dest='face', help='Path of face view video file.', type=str)
    parser.add_argument(
        '--front',dest='front', help='path of front', type=str)
    parser.add_argument(
        '--config',dest='config', help='Path of config file.', type=str)
    parser.add_argument(
        '--output',dest='output', help='Path of output folder.', type=str)
    parser.add_argument(
        '--arch',dest='arch',help='Network architecture, can be: ResNet18, ResNet34, ResNet50, ResNet101, ResNet152',
        default='ResNet50', type=str)

    args = parser.parse_args()
    return args

def getArch(arch,bins):
    # Base network structure
    if arch == 'ResNet18':
        model = L2CS( torchvision.models.resnet.BasicBlock,[2, 2,  2, 2], bins)
    elif arch == 'ResNet34':
        model = L2CS( torchvision.models.resnet.BasicBlock,[3, 4,  6, 3], bins)
    elif arch == 'ResNet101':
        model = L2CS( torchvision.models.resnet.Bottleneck,[3, 4, 23, 3], bins)
    elif arch == 'ResNet152':
        model = L2CS( torchvision.models.resnet.Bottleneck,[3, 8, 36, 3], bins)
    else:
        if arch != 'ResNet50':
            print('Invalid value for architecture is passed! '
                'The default value of ResNet50 will be used instead!')
        model = L2CS( torchvision.models.resnet.Bottleneck, [3, 4, 6,  3], bins)
    return model

def initilize_gaze_model(snapshot_path,arch,batch_size,gpu_id):
    cudnn.enabled = True
    
    transformations = transforms.Compose([
        transforms.Resize(448),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])
    
    model=getArch(arch, 90)
    print('Loading snapshot.')
    torch.cuda.empty_cache()
    model.cuda(gpu)
    saved_state_dict = torch.load(snapshot_path)
    model.load_state_dict(saved_state_dict)
    
    model.eval()

    softmax = nn.Softmax(dim=1)
    detector = RetinaFace(gpu_id=0)
    idx_tensor = [idx for idx in range(90)]
    idx_tensor = torch.FloatTensor(idx_tensor).cuda(gpu)
    
    return model,transformations,softmax,detector,idx_tensor
    
def drawfront(frame_front,front_pitch,front_yaw,pix_x,pix_y,pitch_predicted,yaw_predicted,dist):
    
    R = calibration.Rx(front_pitch)*calibration.Ry(front_yaw)
    #point = np.array([[pix_x],[pix_y],[dist],[1]])-0.082972182-0.236855214
    point = np.array([[pix_x],[pix_y],[dist]])
    #rot_o=R*v1
    #rot_v=calibration.rot_vec(R, pitch_predicted, yaw_predicted)
    #print(rot_v)
    #rotated_R = calibration.Rz(rot_v[0,2]) * calibration.Ry(rot_v[0,1]) * calibration.Rx(rot_v[0,0])
    current_R = calibration.Rx(pitch_predicted)*calibration.Ry(yaw_predicted)
    rotated_R=np.matmul(current_R,R.transpose())
    row_add = np.array([0, 0, 0,1])
    col_add= np.array([0,0,0])
    
    #rotated_R=np.column_stack((rotated_R, col_add))
    #rotated_R=np.vstack ((current_R, row_add) )
    #x,y,z,_=rotated_R.dot(point)
    mapped_point=rotated_R * point
    
    print(mapped_point)
    #img = cv2.circle(frame_front,(int(mapped_point[0,0]),int(mapped_point[1,0])), radius=5, color=(0, 0, 255), thickness=-1)
    return int(mapped_point[0,0]),int(mapped_point[1,0]),frame_front

def maximum(a, b, c):
  
    if (a >= b) and (a >= c):
        largest = a
  
    elif (b >= a) and (b >= c):
        largest = b
    else:
        largest = c
          
    return largest


def eulerToDegrees(euler):
    pi = 22.0/7.0
    return ( (euler) / (2 * pi) ) * 360

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
        print(min(distances))
        return distances[min(distances)]
    else:
        return
                        
    
if __name__ == '__main__':

    args = parse_args()
    gpu_id=args.gpu
    
    
    #get config parameters
    with open(args.config) as stream:
        config = yaml.safe_load(stream)
    
    snapshot_path=config['snapshot_path']
    arch=config['arch']
    batch_size=config['batch_size']
    dist=config['dist']
    f_pitch=config['f_pitch']
    f_yaw=config['f_yaw']
    front_yaw=config['front_yaw'] - f_yaw
    front_pitch=config['front_pitch'] - f_pitch
    
    
    

    #initilize model for gaze detection
    #model,transformations,softmax,detector,idx_tensor=initilize_gaze_model(snapshot_path, arch, batch_size, gpu_id)
    cudnn.enabled = True
    gpu = select_device(str(gpu_id), batch_size=batch_size)
    transformations = transforms.Compose([
        transforms.Resize(448),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])
    
    model=getArch(arch, 90)
    print('Loading snapshot.')
    torch.cuda.empty_cache()
    model.cuda(gpu)
    saved_state_dict = torch.load(snapshot_path)
    model.load_state_dict(saved_state_dict)
    
    model.eval()

    softmax = nn.Softmax(dim=1)
    detector = RetinaFace(gpu_id=0)
    idx_tensor = [idx for idx in range(90)]
    idx_tensor = torch.FloatTensor(idx_tensor).cuda(gpu)
    
    #set filename for the output file
    filename = "l2cs_"+osp.splitext(osp.basename(args.face))[0]
    filename_full = "l2cs_"+osp.basename(args.face)
    front_cap = cv2.VideoCapture(args.front)
    
    #outputFile = osp.join(args.output,filename_full)
    frame_w=int(front_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_h=int(front_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    front_cap.release()
    #capture face view
    face_cap = cv2.VideoCapture(args.face)
    
    #capture front view
    
    #outputFile = osp.join(args.output,filename_full)
    
    
    #initilize the vid writer for wider frame to fit front and face
    fourcc = cv2.VideoWriter_fourcc('M','J','P','G')
   
    i=0
    result=None
    csvout_bbox = open(osp.join(args.output, filename + "_gaze.csv"), "w+")
    print(osp.join(args.output, filename + "_gaze.csv"))
    csvout_bbox.write("frame_no,pitch,yaw,fx_min,fy_min,fx_max,fy_max\n")
    #iterate driver gaze fixation pipeline
    while face_cap.isOpened():
        #get face frame
        ret, frame = face_cap.read()
        #get front frame
        #cv2.imwrite("./img.jpg",frame)
        if ret==True:
            start_fps = time.time()
            #frame = np.rot90(frame,1)
            frame = cv2.resize(frame , (frame_w,frame_h))
            faces = detector(frame)
            if(faces is not None):
                
                for box, landmarks, score in faces:
                    if score < .95:
                        continue
                    x_min=int(box[0])
                    if x_min < 0:
                        x_min = 0
                    y_min=int(box[1])
                    if y_min < 0:
                        y_min = 0
                    x_max=int(box[2])
                    y_max=int(box[3])
                    bbox_width = x_max - x_min
                    bbox_height = y_max - y_min
                    pix_x=(x_max+x_min)//2
                    pix_y=(y_max+y_min)//2 +60
                   
                   
                    # print(i)
                    # Crop image
                    img = frame[y_min:y_max, x_min:x_max]
                    img = cv2.resize(img, (224, 224))
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    im_pil = Image.fromarray(img)
                    img=transformations(im_pil)
                    img  = Variable(img).cuda(gpu)
                    img  = img.unsqueeze(0) 
                    
                    # gaze prediction
                    #print("Running model")
                    gaze_yaw,gaze_pitch = model(img)
                    pitch_predicted = softmax(gaze_pitch)
                    yaw_predicted = softmax(gaze_yaw)
                    

                    # Get continuous predictions in degrees.
                    pitch_predicted = torch.sum(pitch_predicted.data[0] * idx_tensor) * 4 - 180
                    yaw_predicted = torch.sum(yaw_predicted.data[0] * idx_tensor) * 4 - 180
                    
                    pitch_predicted= pitch_predicted.cpu().detach().numpy()* np.pi/180.0
                    yaw_predicted= yaw_predicted.cpu().detach().numpy()* np.pi/180.0
                    gp=eulerToDegrees(pitch_predicted)
                    gy=eulerToDegrees(yaw_predicted)
                    print(str(gp))
                    print(str(gy))
                    #x,y,front=drawfront(frame_front, front_pitch, front_yaw, pix_x, pix_y, pitch_predicted, yaw_predicted, dist)
                    #if(result !=None and len(result)>0):
                    #    closest=closest_object(result, x, y, classes, det_model_threshold)
                    #    if(closest):
                    #        objclass,xmin,ymin,xmax,ymax=closest
                           
                    #else:
                    #     objclass="no object"
                    #     xmin=0
                    #     ymin=0
                    #     xmax=0
                    #     ymax=0
                    csvout_bbox.write('%d,%f,%f,%f,%f,%f,%f' % (i,pitch_predicted,yaw_predicted,x_min,y_min,x_max,y_max) + "\n")
                  
                    

                myFPS = 1.0 / (time.time() - start_fps)
                

                print(frame.shape)
                #print(output.shape)
                print("\n")
                #vid_writer.write(output)
                
                    
                i=i+1
                    
                    #cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0,255,0), 1)
                
            
        else:
            break
        
            
    face_cap.release()
    cv2.destroyAllWindows()