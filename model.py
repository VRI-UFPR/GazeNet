import torch
import torch.nn as nn
from torch.autograd import Variable
import math
import torch.nn.functional as F
import torchvision

class VRI_GazeNet(nn.Module):
    
    def __init__(self, num_bins=181, freeze=False):
        self.freeze = freeze
        self.num_bins = num_bins
        self.binwidth = int(360/(self.num_bins-1))

        super(VRI_GazeNet, self).__init__()
        mobilenet_v2 = torchvision.models.mobilenet_v2(weights='IMAGENET1K_V1')
        self.backbone = mobilenet_v2.features

        if self.freeze:
            for param in self.backbone.parameters():
                param.requires_grad = False

        
        classifier_dict = mobilenet_v2.classifier.state_dict()
        classifier_dict["weight"] = classifier_dict["1.weight"]
        classifier_dict["bias"] = classifier_dict["1.bias"]
        self.fc_yaw_gaze = nn.Sequential(
            nn.Dropout(p=0.3),
            nn.Linear(1280, self.num_bins)
        )

        self.fc_pitch_gaze = nn.Sequential(
            nn.Dropout(p=0.3),
            nn.Linear(1280, self.num_bins)
        )

        try:
            self.fc_yaw_gaze.load_state_dict(classifier_dict)
            self.fc_pitch_gaze.load_state_dict(classifier_dict)
        except RuntimeError as e:
            print(f"IGNORING State dict errors")

        self.softmax = nn.Softmax(dim=1)
        idx_tensor = [idx for idx in range(self.num_bins)]
        self.idx_tensor = torch.FloatTensor(idx_tensor).cpu()
        

    def forward(self, x):
        x = self.backbone(x)
        # straight from https://github.com/pytorch/vision/blob/main/torchvision/models/mobilenetv2.py
        x = nn.functional.adaptive_avg_pool2d(x, (1, 1))
        x = torch.flatten(x, 1)
        # gaze
        pre_yaw_gaze =  self.fc_yaw_gaze(x)
        pre_pitch_gaze = self.fc_pitch_gaze(x)

        yaw = self.softmax(pre_yaw_gaze)
        pitch = self.softmax(pre_pitch_gaze)

        return yaw, pitch


    def angles(self, images):
        y, p = self.forward(images)
        pitch_predicted_cpu = torch.sum(p * self.idx_tensor, 1).cpu().detach().numpy() * self.binwidth - 180
        yaw_predicted_cpu = torch.sum(y * self.idx_tensor, 1).cpu().detach().numpy() * self.binwidth - 180
        return list(zip(yaw_predicted_cpu, pitch_predicted_cpu))

