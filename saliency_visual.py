# -*- coding: utf-8 -*-
"""
Created on Tue Oct 05 15:18:15 2022

@author: Sandesh Jain
Organization: Virginia Tech Transportation Institute
"""

import numpy as np
import matplotlib.pyplot as plt
import cv2
import os
from collections import deque
from matplotlib.lines import Line2D
import argparse

"""
Description: This script considers the CSV file containing the detections and gaze information, 
				the extracted frames, and the beginning and ending indices of the frames to be analyzed
				and produces Yarbus-style sparse point visualization for driver gaze
				The information stack includes the zone of view, the object and the the temporal location of the detection.
"""

def parse_args():
	parser = argparse.ArgumentParser(description='Visual')
	parser.add_argument('--out_dir', type=str, default='./Yarbus_viz/', help='Output directory')
	parser.add_argument('--det_path', type=str, default='./dgf_CMVP_0000_0000_10_130218_1429_00079_Face_adjusted_bbox (1).csv', help='CSV path for the detection and annotations')
	parser.add_argument('--frames_dir', type=str, default='./CHPV_0000_0000_10_130218_1924_00088_Front/', help='Where the extracted frames for your video reside')
	parser.add_argument('--frame_begin', type=str, default=15541, help='Where the frames to be visualized begin (index)')
	parser.add_argument('--frame_end', type=str, default=15840, help='Where the frames to be visualized end (index)')
	
	args = parser.parse_args()
	return args

def matplot(sub_d, figname="part1_adj.png"):
	plt.plot(sub_d[:,1], sub_d[:,2], color='orange')
	plt.plot(sub_d[:,1], sub_d[:,2], 'o', color='black')
	plt.savefig(figname)
	plt.clf()


def free_draw(sub_d, win_size=5, out_dir = './Yarbus_viz_freedraw/', frames_dir = './CHPV_0000_0000_10_130218_1924_00088_Front/'):
	window = deque(maxlen=win_size)
	for row in sub_d:
		fname = frames_dir + str(int(row[0])).zfill(7) + '.jpg'
		x, y = int(row[1]), int(row[2])
		window.append([x,y])
		image = cv2.imread(fname)
		tone = 1
		for pt in window:
			color = tone*255/win_size
			image = cv2.circle(image, (pt[0],pt[1]), radius=4, color=(0, 0, color), thickness=2)
			tone+=1
		cv2.imwrite(out_dir+str(int(row[0])).zfill(7) + '.jpg', image)

def add_text_seq(img, px, py, obj, tx, ty, color):
	# e.g., if the driver has a front-left view on the car:
		# 1.) The view is fixated on X on the front-left segment of the screen
		# 2.) ...
		# Upper left box: keep a tab of the object/ scene where the maximum time is spent (rolling frames)
		
		sector = ''
		
		# three sectors central left and right
		sh = img.shape
		if px>=0 and px<sh[1]/3:
			sector = 'Left'
		elif px>= sh[1]/3 and px<2*sh[1]/3:
			sector ='Central'
		else:
			sector = 'Right'
		
		obj = obj
		
		text = ' Zone: ' + sector + ' Focus: ' + obj
		
		#out_img = cv2.putText(img, text, (tx, ty), cv2.FONT_HERSHEY_SIMPLEX,  0.5, color, 2)

		return text


# Here: take the frame, plot everything in a sequential manner for a specific window size and save them.
def draw_lines(sub_d, win_size=10, out_dir= './Yarbus_viz/', frames_dir = './CHPV_0000_0000_10_130218_1924_00088_Front/'):
	window = deque(maxlen=win_size)
	selected = None
	for row in sub_d:
		fname = frames_dir + str(int(row[0])).zfill(7) + '.jpg'
		im = plt.imread(fname)
		le = []
		sh = im.shape
		x, y, obj = int(row[1]), int(row[2]), str(row[3])
		if not selected:
			tx, ty = sh[1] - 420, 40
			selected = None #True
		ty += 10
		if ty > sh[0]:
			selected = None
		
		text = add_text_seq(im, x, y, obj, tx, ty, color=(0,0,0))
		window.append([x,y, obj, text])
		
		
		fig, ax = plt.subplots()
		fig.set_figwidth(8)

		txt_list = []
		
		im = ax.imshow(im, extent=[0, sh[1], 0, sh[0]])	
		tone = 1
		for (x, y, t) in zip(np.array(window)[:,[0]], np.array(window)[:,[1]], np.array(window)[:,[3]]):
			color = tone/len(window)
			le.append((color,0,1-color))
			txt_list.append(str(t[0]))
			ax.plot(int(x), 356-int(y), 'o', color=(color,0,1-color))
			#ax.text(int(x), 352-int(y), "Point: " + str(tone-1))
			tone+=1
		legend_elements = [Line2D([0], [0], marker='o', color=l, label='Point: '+str(idx) +txt_list[idx],
                          markerfacecolor=l, markersize=5) for idx,l in enumerate(le)]
		tmp = np.array(window)[:,[1]]
		tmp=tmp.astype(int)
		tmp=356-tmp
		tmp2 = np.array(window)[:,[0]]
		tmp2 = tmp2.astype(int)
		ax.plot(tmp2, tmp, color='orange')
		plt.subplots_adjust(right = 0.6, left = 0.04, top = 0.85, bottom = 0.15)
		#plt.figtext(.65, .65, text, fontsize=8)
		ax.legend(handles=legend_elements, bbox_to_anchor=(1.05, 1.05), loc='upper left')
		plt.savefig(out_dir+str(int(row[0])).zfill(7) + '.jpg')
		plt.clf()


def viz(args):
	data_points = np.genfromtxt(args.det_path, delimiter=',', dtype='unicode')
	data_points = data_points.astype('object')
	data_points[:,[0,1,2]] = data_points[:,[0,1,2]].astype(np.int32)
	
	sub_data_row, col = np.where(data_points == args.frame_begin) 
	sub_data_row_end, col = np.where(data_points == args.frame_end) 
	
	sub_d=data_points[sub_data_row[0]:sub_data_row_end[0]]
	if not os.path.isdir(args.out_dir):
		os.makedirs(args.out_dir)
	draw_lines(sub_d, out_dir = args.out_dir,  frames_dir=args.frames_dir)

if __name__ == '__main__':

	args = parse_args()
	viz(args)
