# -*- coding: utf-8 -*-
"""
Created on Wed Aug 31 20:38:18 2022

@author: bhaga
"""

# Copyright (c) Facebook, Inc. and its affiliates.
import cv2
import torch
import atexit
import bisect
import numpy as np
from pyparsing import alphas
import multiprocessing as mp
from collections import deque
import numpy
from detectron2.data import MetadataCatalog
from detectron2.engine.defaults import DefaultPredictor
from detectron2.utils.video_visualizer import VideoVisualizer
from detectron2.utils.visualizer import ColorMode, Visualizer


class VisualizationDemo(object):
    def __init__(self, cfg, instance_mode=ColorMode.IMAGE, parallel=False, overlay=True):
        """
        Args:
            cfg (CfgNode):
            instance_mode (ColorMode):
            parallel (bool): whether to run the model in different processes from visualization.
                Useful since the visualization logic can be slow.
        """
        self.metadata = MetadataCatalog.get(
            cfg.DATASETS.TEST[0] if len(cfg.DATASETS.TEST) else "__unused"
        )
        self.cpu_device = torch.device("cpu")
        self.instance_mode = instance_mode

        self.parallel = parallel
        if parallel:
            num_gpu = torch.cuda.device_count()
            self.predictor = AsyncPredictor(cfg, num_gpus=num_gpu)
        else:
            self.predictor = DefaultPredictor(cfg)

        self.overlay = overlay
    




    def run_on_image(self, image,g_x,g_y):
        """
        Args:
            image (np.ndarray): an image of shape (H, W, C) (in BGR order).
                This is the format used by OpenCV.

        Returns:
            predictions (dict): the output of the model.
            vis_output (VisImage): the visualized image output.
        """
        vis_output = None
        predictions = self.predictor(image)
        # Convert image from OpenCV BGR format to Matplotlib RGB format.
        image = image[:, :, ::-1]
        #print("here")
        if not self.overlay:
            visualizer = Visualizer(
                np.zeros(image.shape), self.metadata, instance_mode=self.instance_mode)

            if "panoptic_seg" in predictions:
                panoptic_seg, segments_info = predictions["panoptic_seg"]
                #print(type(panoptic_seg))
                vis_output = visualizer.draw_panoptic_seg_predictions(
                    panoptic_seg.to(self.cpu_device), segments_info, alpha=1.0
                )
            else:
                if "sem_seg" in predictions:
                    vis_output = visualizer.draw_sem_seg(
                        predictions["sem_seg"].argmax(
                            dim=0).to(self.cpu_device), alpha=1.0
                    )
                if "instances" in predictions:
                    instances = predictions["instances"].to(self.cpu_device)
                    vis_output = visualizer.draw_instance_predictions(
                        predictions=instances)
        else:
            visualizer = Visualizer(
                image, self.metadata, instance_mode=self.instance_mode)

            if "panoptic_seg" in predictions:
                panoptic_seg, segments_info = predictions["panoptic_seg"]
                ps=panoptic_seg.to(self.cpu_device)
                ps=ps.numpy()
                cat_id=ps[g_y,g_x]
                mask = np.zeros(ps.shape, np.uint8)
                cv2.circle(mask, (g_x,g_y), 30, 255, -1)
                where = np.where(mask == 255)
                intensity_values_from_original = ps[where[0],where[1]]
                #print(intensity_values_from_original)
                unique, counts = numpy.unique(intensity_values_from_original, return_counts=True) 
                i_c=dict(zip(unique, counts))
                ids=[]
                labels=[]
                label_id="N/A"
                class_names=self.metadata.thing_classes
                #print(x["isthing"])
                for x in segments_info:
                    #print(x["isthing"])
                    if(x["isthing"]==True and x["id"] in i_c.keys()):
                       ids.append(x["id"])
                       #if(x["category_id"]==149):
                       #     print("ROAD")
                       #if(class_names[x["category_id"]] =="road"):
                       #     print(x["category_id"])
                       #     print(cat_id)
                       #     break
                       if(x["instance_id"]==cat_id):
                            label_id=x["category_id"]
                       labels.append(x["category_id"])
                    else:
                        ps[ps == x["id"]] = 0                       
                       
                if(len(ids)>0):
                    m=0
                    new_i=0
                    #print(i_c)
                    #print(ids)
                    #print(type(ids[0]))
                    for i,l in zip(ids,labels):
                       if(i_c[i]>m):
                           new_i=i
                           m=i_c[i]
                           label_id=l
                    ps[ps != new_i] = 0
                    # a = np.indices(ps.shape).reshape(2,-1).T
                    # circ=a[np.abs(a-np.array([g_x,g_y])).sum(1)<=int(40)]-np.array([g_x,g_y])
                    # difference_array = np.absolute(ps-i)
      
                    # # find the index of minimum element from the array
                    # index = np.unravel_index(np.argmin(difference_array, axis=None), difference_array.shape)
                    # ind = np.unravel_index(np.argmax(difference_array, axis=None), difference_array.shape)
                    # mid_index=(ind[0]+index[0])//2,(ind[1]+index[1])//2
                    if(new_i!=0):
                    	mass_x, mass_y = np.where(ps == new_i)
                    # mass_x and mass_y are the list of x indices and y indices of mass pixels
                    
                    	cent_x = np.average(mass_x)
                    	cent_y = np.average(mass_y)
                    	index=int(cent_y),int(cent_x)
                    else:
                        cat_id=0
                        ps[ps != cat_id] = 0
                        index=g_x,g_y
                    
                    
                else:
                    cat_id=0
                    ps[ps != cat_id] = 0
                    index=g_x,g_y
                
                #print(ps.shape)
                #panoptic_seg.to(self.cpu_device)
                
                #category_idx = segments_info[cat_id]
                #print(class_names[label_id])
                if(label_id!="N/A"):
                	label=class_names[label_id]
                else:
                        label="N/A"
                #print(label)
                vis_output = visualizer.draw_panoptic_seg_predictions(
                    torch.from_numpy(ps), segments_info
                )
            else:
                if "sem_seg" in predictions:
                    vis_output = visualizer.draw_sem_seg(
                        predictions["sem_seg"].argmax(
                            dim=0).to(self.cpu_device)
                    )
                if "instances" in predictions:
                    instances = predictions["instances"].to(self.cpu_device)
                    vis_output = visualizer.draw_instance_predictions(
                        predictions=instances)

        return predictions, vis_output,index,label

    def _frame_from_video(self, video):
        while video.isOpened():
            success, frame = video.read()
            if success:
                yield frame
            else:
                break

    def run_on_video(self, video):
        """
        Visualizes predictions on frames of the input video.

        Args:
            video (cv2.VideoCapture): a :class:`VideoCapture` object, whose source can be
                either a webcam or a video file.

        Yields:
            ndarray: BGR visualizations of each video frame.
        """
        video_visualizer = VideoVisualizer(self.metadata, self.instance_mode)

        def process_predictions(frame, predictions):
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            if not self.overlay:
                if "panoptic_seg" in predictions:
                    panoptic_seg, segments_info = predictions["panoptic_seg"]
                    vis_frame = video_visualizer.draw_panoptic_seg_predictions(np.zeros(
                        frame.shape), panoptic_seg.to(self.cpu_device), segments_info, alpha=1.0)

                elif "instances" in predictions:
                    predictions = predictions["instances"].to(self.cpu_device)
                    vis_frame = video_visualizer.draw_instance_predictions(
                        frame, predictions)
                elif "sem_seg" in predictions:
                    vis_frame = video_visualizer.draw_sem_seg(
                        frame, predictions["sem_seg"].argmax(
                            dim=0).to(self.cpu_device)
                    )
            else:
                if "panoptic_seg" in predictions:
                    panoptic_seg, segments_info = predictions["panoptic_seg"]
                    vis_frame = video_visualizer.draw_panoptic_seg_predictions(
                        frame, panoptic_seg.to(self.cpu_device), segments_info)

                elif "instances" in predictions:
                    predictions = predictions["instances"].to(self.cpu_device)
                    vis_frame = video_visualizer.draw_instance_predictions(
                        frame, predictions)
                elif "sem_seg" in predictions:
                    vis_frame = video_visualizer.draw_sem_seg(
                        frame, predictions["sem_seg"].argmax(
                            dim=0).to(self.cpu_device)
                    )

            # Converts Matplotlib RGB format to OpenCV BGR format
            vis_frame = cv2.cvtColor(vis_frame.get_image(), cv2.COLOR_RGB2BGR)
            return predictions, vis_frame

        frame_gen = self._frame_from_video(video)
        if self.parallel:
            buffer_size = self.predictor.default_buffer_size

            frame_data = deque()

            for cnt, frame in enumerate(frame_gen):
                frame_data.append(frame)
                self.predictor.put(frame)

                if cnt >= buffer_size:
                    frame = frame_data.popleft()
                    predictions = self.predictor.get()
                    yield process_predictions(frame, predictions)

            while len(frame_data):
                frame = frame_data.popleft()
                predictions = self.predictor.get()
                yield process_predictions(frame, predictions)
        else:
            for frame in frame_gen:
                yield process_predictions(frame, self.predictor(frame))


class AsyncPredictor:
    """
    A predictor that runs the model asynchronously, possibly on >1 GPUs.
    Because rendering the visualization takes considerably amount of time,
    this helps improve throughput a little bit when rendering videos.
    """

    class _StopToken:
        pass

    class _PredictWorker(mp.Process):
        def __init__(self, cfg, task_queue, result_queue):
            self.cfg = cfg
            self.task_queue = task_queue
            self.result_queue = result_queue
            super().__init__()

        def run(self):
            predictor = DefaultPredictor(self.cfg)

            while True:
                task = self.task_queue.get()
                if isinstance(task, AsyncPredictor._StopToken):
                    break
                idx, data = task
                result = predictor(data)
                self.result_queue.put((idx, result))

    def __init__(self, cfg, num_gpus: int = 1):
        """
        Args:
            cfg (CfgNode):
            num_gpus (int): if 0, will run on CPU
        """
        num_workers = max(num_gpus, 1)
        self.task_queue = mp.Queue(maxsize=num_workers * 3)
        self.result_queue = mp.Queue(maxsize=num_workers * 3)
        self.procs = []
        for gpuid in range(max(num_gpus, 1)):
            cfg = cfg.clone()
            cfg.defrost()
            cfg.MODEL.DEVICE = "cuda:{}".format(
                gpuid) if num_gpus > 0 else "cpu"
            self.procs.append(
                AsyncPredictor._PredictWorker(
                    cfg, self.task_queue, self.result_queue)
            )

        self.put_idx = 0
        self.get_idx = 0
        self.result_rank = []
        self.result_data = []

        for p in self.procs:
            p.start()
        atexit.register(self.shutdown)

    def put(self, image):
        self.put_idx += 1
        self.task_queue.put((self.put_idx, image))

    def get(self):
        self.get_idx += 1  # the index needed for this request
        if len(self.result_rank) and self.result_rank[0] == self.get_idx:
            res = self.result_data[0]
            del self.result_data[0], self.result_rank[0]
            return res

        while True:
            # make sure the results are returned in the correct order
            idx, res = self.result_queue.get()
            if idx == self.get_idx:
                return res
            insert = bisect.bisect(self.result_rank, idx)
            self.result_rank.insert(insert, idx)
            self.result_data.insert(insert, res)

    def __len__(self):
        return self.put_idx - self.get_idx

    def __call__(self, image):
        self.put(image)
        return self.get()

    def shutdown(self):
        for _ in self.procs:
            self.task_queue.put(AsyncPredictor._StopToken())

    @ property
    def default_buffer_size(self):
        return len(self.procs) * 5
