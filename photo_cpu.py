import argparse
import numpy as np
import cv2
import time

import torch
import torch.nn as nn
from torch.autograd import Variable
from torchvision import transforms
import torch.backends.cudnn as cudnn
import torchvision

from PIL import Image
from utils import select_device, draw_gaze
from PIL import Image, ImageOps

from face_detection import RetinaFace
# from model import ML2CS180
from model import VRI_GazeNet
import os

def parse_args():
    """Parse input arguments."""
    parser = argparse.ArgumentParser(
        description='Gaze evalution using model pretrained with L2CS-Net on Gaze360.')
    parser.add_argument(
        '--gpu',dest='gpu_id', help='GPU device id to use [0]',
        default="0", type=str)
    parser.add_argument(
        '--snapshot',dest='snapshot', help='Path of model snapshot.', 
        default='models/VRI-GazeNet.pkl', type=str)
        

    parser.add_argument(
        '--image',dest='image_filename', help='Image', type=str)

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

if __name__ == '__main__':
    args = parse_args()

    cudnn.enabled = True

    image_filename = args.image_filename

    batch_size = 1
    # cam = args.cam_id
    gpu = select_device(args.gpu_id, batch_size=batch_size)
    snapshot_path = args.snapshot

    transformations = transforms.Compose([
        transforms.Resize(224),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])
    
    model = VRI_GazeNet(num_bins=181)
    print('Loading snapshot.')
    saved_state_dict = torch.load(snapshot_path, map_location=torch.device('cpu'))
    model.load_state_dict(saved_state_dict)
    model.cpu()
    model.eval()

    softmax = nn.Softmax(dim=1)
    detector = RetinaFace(gpu_id=-1)

    # idx_tensor = torch.FloatTensor(idx_tensor).cuda(gpu)
    idx_tensor = [idx for idx in range(model.num_bins)]
    idx_tensor = torch.FloatTensor(idx_tensor).cpu()


    if os.path.isdir(image_filename):
        frames = []
        for file in os.listdir(image_filename):
            frame = cv2.imread(os.path.join(image_filename, file))

            if frame is not None:
                frames.append((frame, os.path.basename(file)))
            
    else:
        frame = cv2.imread(image_filename)
        frames = [(frame, os.path.basename(image_filename))]


    with torch.no_grad():

        for frame, filename in frames:
        
            faces = detector(frame)
            if faces is not None: 
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
                    # Crop image
                    img = frame[y_min:y_max, x_min:x_max]
                    img = cv2.resize(img, (224, 224))
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    im_pil = Image.fromarray(img)
                    img=transformations(im_pil)
                    # img  = Variable(img).cuda(gpu)
                    img  = Variable(img).cpu()
                    img  = img.unsqueeze(0) 
                    
                    # gaze prediction
                    gazes = model.angles(img)
                    yaw, pitch = gazes[0]
                    print(yaw, pitch)
                    
                    # pitch_predicted= gaze_pitch.cpu().detach().numpy()* np.pi/180.0
                    # yaw_predicted= gaze_yaw.cpu().detach().numpy()* np.pi/180.0
                    yaw_predicted= yaw * np.pi/180.0
                    pitch_predicted= pitch * np.pi/180.0
                    
                    # cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (10,200,90), 1)
                    # draw_gaze(x_min,y_min,bbox_width, bbox_height,frame,(yaw_predicted,pitch_predicted),color=(150, 100, 92), scale=1, thickness=4, size=x_max-x_min, bbox=((x_min, y_min), (x_max, y_max)))
                    draw_gaze(x_min,y_min,bbox_width, bbox_height,frame,(yaw_predicted,pitch_predicted),color=(185, 240, 113), scale=1, thickness=4, size=x_max-x_min, bbox=((x_min, y_min), (x_max, y_max)))
                    # 
                    # cv2.putText(frame, f"{pitch_predicted, yaw_predicted}", (x_min,y_min), cv2.FONT_HERSHEY_PLAIN, 2, (255, 255, 255), 2)

            cv2.imwrite("gaze_"+filename, frame)