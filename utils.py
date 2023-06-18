import numpy as np
import torch
import torch.nn as nn
import os
import scipy.io as sio
import cv2
import math
from math import cos, sin
from pathlib import Path
import subprocess
import re
from model import L2CS
import torchvision
import sys


def atoi(text):
    return int(text) if text.isdigit() else text

def natural_keys(text):
    '''
    alist.sort(key=natural_keys) sorts in human order
    http://nedbatchelder.com/blog/200712/human_sorting.html
    (See Toothy's implementation in the comments)
    '''
    return [ atoi(c) for c in re.split(r'(\d+)', text) ]

def gazeto3d(gaze):
  gaze_gt = np.zeros([3])
  gaze_gt[0] = -np.cos(gaze[1]) * np.sin(gaze[0])
  gaze_gt[1] = -np.sin(gaze[1])
  gaze_gt[2] = -np.cos(gaze[1]) * np.cos(gaze[0])
  return gaze_gt

def angular(gaze, label):
  total = np.sum(gaze * label)
  return np.arccos(min(total/(np.linalg.norm(gaze)* np.linalg.norm(label)), 0.9999999))*180/np.pi

def angular_torch(gaze, label):
  total = torch.sum(gaze * label)
  return np.arccos(min(total/(np.linalg.norm(gaze)* np.linalg.norm(label)), 0.9999999))*180/np.pi


def draw_gaze(a,b,c,d,image_in, yawpitch, thickness=5, color=(255, 255, 255),scale=2.0, size=0, bbox=None, tip=True):
    """Draw gaze angle on given image with a given eye positions."""

    # print(f"Drawing {yawpitch}")

    thickness = int(thickness * size/85)
    if thickness < 1:
        thickness = 1

    image_out = image_in
    (h, w) = image_in.shape[:2]
    length = size/2 * scale
    if len(image_out.shape) == 2 or image_out.shape[2] == 1:
        image_out = cv2.cvtColor(image_out, cv2.COLOR_GRAY2BGR)

    dx = -length * np.sin(yawpitch[0]) * np.cos(yawpitch[1])
    dy = -length * np.sin(yawpitch[1])
    pos = [int(a+c / 2.0), int( b+d / 3.5)]
    # pos = [int(a+c / 2.0) + dx//2, int( b+d / 3.5) + dy//2]

    if pos[0] > bbox[1][0]:
        pos[0] = bbox[1][0]
    elif pos[0] < bbox[0][0]:
        pos[0] = bbox[0][0]

    if pos[1] > bbox[1][1]:
        pos[1] = bbox[1][1]
    elif pos[1] < bbox[0][1]:
        pos[1] = bbox[0][1]

    cv2.arrowedLine(image_out, tuple(np.round(pos).astype(np.int32)),
                   tuple(np.round([pos[0] + dx, pos[1] + dy]).astype(int)), color,
                   thickness, cv2.LINE_AA, tipLength=0.1)
    
    color2 = (color[0]+20, color[1]+20, color[2]+20)
    # color3 = (0, 0, 0)
    color3 = (color[0]-80, color[1]-80, color[2]-80)
    # color3 = (color[0]-40, color[1]-40, color[2]-40)



    if tip and (bbox[0][0] < pos[0] + dx < bbox[1][0]) and (bbox[0][1] < pos[1] + dy < bbox[1][1]):
        cv2.circle(image_out, (int(pos[0] + dx), int(pos[1] + dy)), int(thickness*1.4), color2, -1)
        cv2.circle(image_out, (int(pos[0] + dx*1.1), int(pos[1] + dy*1.1)), 1, color3, 2)
        cv2.circle(image_out, (int(pos[0] + dx), int(pos[1] + dy)), int(thickness*1.4), color3, 1)


    return image_out    

def select_device(device='', batch_size=None):
    # device = 'cpu' or '0' or '0,1,2,3'
    s = f'YOLOv3 🚀 {git_describe() or date_modified()} torch {torch.__version__} '  # string
    cpu = device.lower() == 'cpu'
    if cpu:
        os.environ['CUDA_VISIBLE_DEVICES'] = '-1'  # force torch.cuda.is_available() = False
    elif device:  # non-cpu device requested
        os.environ['CUDA_VISIBLE_DEVICES'] = device  # set environment variable
        assert torch.cuda.is_available(), f'CUDA unavailable, invalid device {device} requested'  # check availability

    cuda = not cpu and torch.cuda.is_available()
    if cuda:
        devices = device.split(',') if device else range(torch.cuda.device_count())  # i.e. 0,1,6,7
        n = len(devices)  # device count
        if n > 1 and batch_size:  # check batch_size is divisible by device_count
            assert batch_size % n == 0, f'batch-size {batch_size} not multiple of GPU count {n}'
        space = ' ' * len(s)
        for i, d in enumerate(devices):
            p = torch.cuda.get_device_properties(i)
            s += f"{'' if i == 0 else space}CUDA:{d} ({p.name}, {p.total_memory / 1024 ** 2}MB)\n"  # bytes to MB
    else:
        s += 'CPU\n'

    return torch.device('cuda:0' if cuda else 'cpu')

def spherical2cartesial(x):
    
    output = torch.zeros(x.size(0),3)
    output[:,2] = -torch.cos(x[:,1])*torch.cos(x[:,0])
    output[:,0] = torch.cos(x[:,1])*torch.sin(x[:,0])
    output[:,1] = torch.sin(x[:,1])

    return output
    
def compute_angular_error(input,target):

    input = spherical2cartesial(input)
    target = spherical2cartesial(target)

    input = input.view(-1,3,1)
    target = target.view(-1,1,3)
    output_dot = torch.bmm(target,input)
    output_dot = output_dot.view(-1)
    output_dot = torch.acos(output_dot)
    output_dot = output_dot.data
    output_dot = 180*torch.mean(output_dot)/math.pi
    return output_dot

def softmax_temperature(tensor, temperature):
    result = torch.exp(tensor / temperature)
    result = torch.div(result, torch.sum(result, 1).unsqueeze(1).expand_as(result))
    return result
   
def git_describe(path=Path(__file__).parent):  # path must be a directory
    # return human-readable git description, i.e. v5.0-5-g3e25f1e https://git-scm.com/docs/git-describe
    s = f'git -C {path} describe --tags --long --always'
    try:
        return subprocess.check_output(s, shell=True, stderr=subprocess.STDOUT).decode()[:-1]
    except subprocess.CalledProcessError as e:
        return ''  # not a git repository

