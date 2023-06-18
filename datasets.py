import os
import numpy as np
import cv2


import torch
from torch.utils.data.dataset import Dataset
from torchvision import transforms
from PIL import Image, ImageFilter, ImageFile, ImageOps
ImageFile.LOAD_TRUNCATED_IMAGES = True


class Gaze360(Dataset):
    def __init__(self, path, root, transform, angle, binwidth, num_bins, train=True):
        self.num_bins = num_bins
        self.transform = transform
        self.root = root
        self.orig_list_len = 0
        self.angle = angle
        self.binwidth=binwidth
        self.lines = []
        if isinstance(path, list):
            for i in path:
                with open(i) as f:
                    print("here")
                    line = f.readlines()
                    line.pop(0)
                    self.lines.extend(line)
        else:
            print("WILL FILTER OOOOOOO")
            with open(path) as f:
                lines = f.readlines()
                lines.pop(0)
                self.orig_list_len = len(lines)
                for line in lines:
                    gaze2d = line.strip().split(" ")[5]
                    label = np.array(gaze2d.split(",")).astype("float")
                    if abs((label[0]*180/np.pi)) <= angle and abs((label[1]*180/np.pi)) <= angle:
                        self.lines.append(line)
                    
                        
        print("{} items removed from dataset that have an angle > {}".format(self.orig_list_len-len(self.lines), angle))
        print(self.orig_list_len, len(self.lines))

    def __len__(self):
        return len(self.lines)

    def __getitem__(self, idx):
        line = self.lines[idx]
        line = line.strip().split(" ")

        face = line[0]
        lefteye = line[1]
        righteye = line[2]
        name = line[3]
        gaze2d = line[5]
        label = np.array(gaze2d.split(",")).astype("float")
        label = torch.from_numpy(label).type(torch.FloatTensor)

        yaw = label[0]* 180 / np.pi
        pitch = label[1]* 180 / np.pi

        img = Image.open(os.path.join(self.root, face))

        if self.transform:
            img = self.transform(img)
                
        
        # Bin values
        # bins = np.array(range(-1*self.angle, self.angle, self.binwidth))
        # print(self.num_bins)
        bins = np.array(range(-1*self.num_bins, self.num_bins + self.binwidth, self.binwidth))
        # print(bins)
        binned_pose = np.digitize([yaw, pitch], bins)

        # print(binned_pose)
        labels = binned_pose
        cont_labels = torch.FloatTensor([yaw, pitch])

        return img, labels, cont_labels, name


class Mpiigaze(Dataset): 
  def __init__(self, pathorg, root, transform,angle, binwidth, num_bins, fold=0):
    self.num_bins = num_bins
    
    self.transform = transform
    self.root = root
    self.orig_list_len = 0
    self.lines = []
    # path=pathorg.copy()
    path=pathorg
    # if train==True:
    #   path.pop(fold)
    # else:
    #   path=path[fold]
    self.binwidth=binwidth
    self.angle = angle


    # if isinstance(path, list):
    if isinstance(path, str) and os.path.isdir(path):
        folder = os.listdir(path)
        folder.sort()
        folder = [os.path.join(path, j) for j in folder]
        for i in folder:
            with open(i) as f:
                lines = f.readlines()
                lines.pop(0)
                self.orig_list_len += len(lines)
                for line in lines:
                    gaze2d = line.strip().split(" ")[7]
                    label = np.array(gaze2d.split(",")).astype("float")
                    if abs((label[0]*180/np.pi)) <= angle and abs((label[1]*180/np.pi)) <= angle:
                        self.lines.append(line)
    else:
      with open(path) as f:
        lines = f.readlines()
        lines.pop(0)
        self.orig_list_len += len(lines)
        for line in lines:
            gaze2d = line.strip().split(" ")[7]
            label = np.array(gaze2d.split(",")).astype("float")
            if abs((label[0]*180/np.pi)) <= angle and abs((label[1]*180/np.pi)) <= angle:
                self.lines.append(line)
   
    print("{} items removed from dataset that have an angle > {}".format(self.orig_list_len-len(self.lines),angle))
    print(self.orig_list_len, len(self.lines))

        
  def __len__(self):
    return len(self.lines)

  def __getitem__(self, idx):
    line = self.lines[idx]
    line = line.strip().split(" ")

    name = line[3]
    gaze2d = line[7]
    head2d = line[8]
    lefteye = line[1]
    righteye = line[2]
    face = line[0]

    label = np.array(gaze2d.split(",")).astype("float")
    label = torch.from_numpy(label).type(torch.FloatTensor)


    yaw = label[0]* 180 / np.pi
    pitch = label[1]* 180 / np.pi

    img = Image.open(os.path.join(self.root, face))
    
    if self.transform:
        img = self.transform(img)        
    
    # Bin values
    # bins = np.array(range(-1*self.angle, self.angle, self.binwidth))
    # print(self.num_bins)
    # bins = np.array(range(-1*self.num_bins//2, self.num_bins//2 + self.binwidth, self.binwidth))
    bins = np.array(range(-1*self.num_bins, self.num_bins + self.binwidth, self.binwidth))

    # print(bins)

    binned_pose = np.digitize([yaw, pitch], bins)
    # print(binned_pose)


    labels = binned_pose
    cont_labels = torch.FloatTensor([yaw, pitch])

    return img, labels, cont_labels, name