import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision import transforms
import torch.backends.cudnn as cudnn
import torchvision

import datasets
from utils import select_device, natural_keys, gazeto3d, angular
from model import L2CS, VRI_GazeNet

from fvcore.nn import FlopCountAnalysis
import typing
import datetime

def parse_args():
    """Parse input arguments."""
    parser = argparse.ArgumentParser(
        description='Gaze estimation using L2CSNet .')
     # Gaze360
    parser.add_argument(
        '--gaze360image_dir_test', dest='gaze360image_dir_test', help='Directory path for gaze images.',
        default='../gaze360_test/Image', type=str)
    parser.add_argument(
        '--gaze360label_dir_test', dest='gaze360label_dir_test', help='Directory path for gaze labels.',
        default='../gaze360_test/Label', type=str)
   
    # Important args -------------------------------------------------------------------------------------------------------
    # ----------------------------------------------------------------------------------------------------------------------
    parser.add_argument(
        '--dataset', dest='dataset', help='gaze360, mpiigaze',
        default= "gaze360", type=str)
    parser.add_argument(
        '--snapshot', dest='snapshot', help='Path to the folder contains models.', 
        default='output/snapshots/L2CS-gaze360-_loader-180-4-lr', type=str)
    parser.add_argument(
        '--evalpath', dest='evalpath', help='path for the output evaluating gaze test.',
        default="evaluation/L2CS-gaze360-_loader-180-4-lr", type=str)
    parser.add_argument(
        '--gpu',dest='gpu_id', help='GPU device id to use [0]',
        default="0", type=str)
    parser.add_argument(
        '--batch_size', dest='batch_size', help='Batch size.',
        default=100, type=int)
    parser.add_argument(
        '--arch', dest='arch', help='Network architecture, can be: ResNet18, ResNet34, [ResNet50], ''ResNet101, ResNet152, Squeezenet_1_0, Squeezenet_1_1, MobileNetV2',
        default='ResNet50', type=str)
    parser.add_argument(
    '--angle', dest='angle', help='bruh', default=90, type=int)
    # ---------------------------------------------------------------------------------------------------------------------
    # Important args ------------------------------------------------------------------------------------------------------
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()
    cudnn.enabled = True
    gpu = select_device(args.gpu_id, batch_size=args.batch_size)
    batch_size=args.batch_size
    angle=args.angle

    transformations = transforms.Compose([
        transforms.Resize(224),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])

    vri = VRI_GazeNet()
    vri.name = "VRI"
    saved_state_dict = torch.load("../models/VRI-180-May-27-LR1e-05-DEC1e-06-drop0.3-BATCH8-augment-CROSSENTROPY-False-11.49t360-11.22t180.pkl")
    vri.load_state_dict(saved_state_dict)

    l2cs = L2CS(torchvision.models.resnet.Bottleneck, [3, 4, 6, 3], 90)
    l2cs.num_bins = 90
    l2cs.name = "L2CS"
    saved_state_dict = torch.load("../models/L2CSNet_gaze360.pkl")
    l2cs.load_state_dict(saved_state_dict)

    # TEST
    folder = os.listdir(args.gaze360label_dir_test)
    folder.sort()
    testlabelpathombined = [os.path.join(args.gaze360label_dir_test, j) for j in folder]
    gaze_dataset_test=datasets.Gaze360(testlabelpathombined,args.gaze360image_dir_test, transformations, 360, vri.binwidth)
    
    test_loader = torch.utils.data.DataLoader(
        dataset=gaze_dataset_test,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True)

    for model in (l2cs, vri):
        # Base network structure
        model.cuda(gpu)
        model.eval()
        total = 0
        start = datetime.datetime.now()
        with torch.no_grad():           
            for j, (images, labels, cont_labels, name) in enumerate(test_loader):
                images = Variable(images).cuda(gpu)
                yaw_predicted, pitch_predicted = model(images)    
                total += cont_labels.size(0)

        end = datetime.datetime.now()
        duration = end - start
        seconds = duration.total_seconds()
                
        log = f"[{model.name}] Images:{total}. Duration:{seconds}. FPS = {total/seconds}"
        print(log)
