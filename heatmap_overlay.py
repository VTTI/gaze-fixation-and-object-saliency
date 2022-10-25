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
    
def myplot(x, y, s,dim):
    heatmap, xedges, yedges = np.histogram2d(x, y, bins=dim)
    heatmap = gaussian_filter(heatmap, sigma=s)

    #extent = [0, 480, 0, 360]
    return heatmap.T

face=str(sys.argv[1])
front=str(sys.argv[2])
output_name=str(sys.argv[3])
config=str(sys.argv[4])
trail_id=str(sys.argv[5])

face_cap = cv2.VideoCapture(face)
front_cap = cv2.VideoCapture(front)
print(front_cap.isOpened())
fps=14
filename = "dgf"+osp.splitext(osp.basename(face))[0]
filename_full = "dgf_"+osp.basename(face)
outputFile = osp.join(output_name,filename_full)
frame_w=int(front_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_h=int(front_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
filename = "l2cs_"+osp.splitext(face)[0]
csv_file=osp.join(output_name,"filtered_dgf_l2cs_"+osp.splitext(osp.basename(face))[0] + "_bbox.csv")
csv_file2=osp.join(output_name,"Objdata_"+osp.splitext(osp.basename(face))[0] + "_bbox.csv")
#initilize the vid writer for wider frame to fit front and face
fourcc = cv2.VideoWriter_fourcc('M','J','P','G')
vid_writer = cv2.VideoWriter(outputFile, 0x7634706d, int(face_cap.get(cv2.CAP_PROP_FPS)),
                                (int(front_cap.get(cv2.CAP_PROP_FRAME_WIDTH))*2, int(front_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))))

output = np.zeros((frame_h, frame_w * 2, 3), dtype="uint8")

with open(config) as stream:
    config = yaml.safe_load(stream)

f_pitch=config['f_pitch']
f_yaw=config['f_yaw']
dist=config['dist']
front_yaw=config['front_yaw'] - f_yaw
front_pitch=config['front_pitch'] - f_pitch
det_model_config=config['det_model_config']
det_model_classes=config['det_model_classes']
det_model_checkpoint=config['det_model_checkpoint']
det_model_threshold=config['det_model_threshold']
det_model_id=config['det_gpu_id']
print(det_model_checkpoint)
det_model = init_detector(det_model_config, det_model_checkpoint, device='cuda:'+str(det_model_id))

with open(csv_file) as fd:
    reader = fd.readlines()
obj_bbox = open(csv_file2, "w+")
obj_bbox.write("trail_id,frame_no,obj_class\n")

i=1
j=0
slide_x=[]
slide_y=[]
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
                frame_no=int(float(row[0]))
                print(j)
                print(frame_no)
                if(j==frame_no):
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
                    fy=(frame_h)//2 +60
                    
                    pix_x,pix_y,front=drawfront(frame_front, front_pitch, front_yaw, fx, fy, pitch, yaw, dist)
                    if(pix_y>frame_h-60):
                    	pix_y=pix_y+60
                    
                   
                    p=m.degrees(pitch)
                    y=m.degrees(yaw)
                    cv2.putText(frame, 'Pitch: {:.1f}, Yaw:{:.1f}'.format(p,-y), (10, 20),cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 255, 0), 1, cv2.LINE_AA)
                    draw_gaze(fx_min,fy_min,bbox_width, bbox_height,frame,(yaw,pitch),color=(0,0,255))
                    front = cv2.circle(frame_front,(pix_x,pix_y), radius=5, color=(0, 0, 255), thickness=-1)
                    #print(result)
                    if(len(result)>0):
                        closest=closest_object(result, pix_x, pix_y, classes, det_model_threshold)
                        if(closest):
                            objclass,xmin,ymin,xmax,ymax=closest
                            #csvout_bbox.write(str(i) + ","+objclass + "," +'%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f' % (xmin,ymin,xmax,ymax,x,y,pitch_predicted,yaw_predicted,x_min,y_min,x_max,y_max) + "\n")
                            obj_bbox.write(str(trail_id)+","+str(frame_no)+","+objclass+"\n")
                            front=cv2.rectangle(front, (int(xmin),int(ymin)), (int(xmax),int(ymax)),(0, 255, 0), 2)
                            front = cv2.putText(front, objclass, (int(xmin), int(ymin)-5), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 255, 0), 1)
                    
                    if(pix_x<frame_w and pix_y<frame_h):
                        
                        slide_x.append(pix_x)
                        slide_x.append(pix_x)
                        slide_x.append(pix_x)
                        slide_x.append(pix_x)
                        slide_y.append(pix_y)
                        slide_y.append(pix_y)
                        slide_y.append(pix_y)
                        slide_y.append(pix_y)
                    else:
                        cv2.putText(front, 'Gaze out of frame', (10, 20),cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 255, 0), 1, cv2.LINE_AA)
                        
                    if(len(slide_x)>=record_heatmap):
                        
                        img = myplot(slide_x, slide_y, 64,[frame_w,frame_h])
                        plt.imsave("./test1.png", img,format="png", cmap=cm.jet)
                        read_img=cv2.imread("./test1.png")
                        slide_x=slide_x[1:]
                        slide_y=slide_y[1:]
                        print(read_img.shape)
                        print(front.shape)
                        front=cv2.addWeighted(read_img, 0.5, front, 0.5, 0)
                        slide_x=slide_x[:-3]
                        slide_y=slide_y[:-3]
                    output[0:frame_h, frame_w:frame_w*2]=frame
                    output[0:frame_h, 0:frame_w]=front
                    vid_writer.write(output)
                else:
                    output[0:frame_h, frame_w:frame_w*2]=frame
                    output[0:frame_h, 0:frame_w]=frame_front
                    vid_writer.write(output)
                j=j+1

                
            else:
                break
            
                
face_cap.release()
front_cap.release()
cv2.destroyAllWindows()

        