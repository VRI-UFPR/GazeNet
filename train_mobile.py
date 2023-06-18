import os
import argparse
import time

import torch.utils.model_zoo as model_zoo
import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.utils.data import DataLoader, ConcatDataset
from torchvision import transforms
import torch.backends.cudnn as cudnn
import torchvision

import datasets
# from model import ML2CS, ML2CS180
from model import VRI_GazeNet
from utils import select_device, natural_keys, gazeto3d, angular
import numpy as np

import matplotlib.pyplot as plt
from datetime import datetime
from calendar import month_name

from PIL import ImageOps


def parse_args():
    """Parse input arguments."""
    
    # DATASET ARGS
    parser = argparse.ArgumentParser(description='Gaze estimation using L2CSNet.')
    parser.add_argument(
        '--gaze360image_dir_train', dest='gaze360image_dir_train', help='Directory path for gaze images.',
        default='../gaze360_train/Image', type=str)
    parser.add_argument(
        '--gaze360label_dir_train', dest='gaze360label_dir_train', help='Directory path for gaze labels.',
        default='../gaze360_train/Label', type=str)
    parser.add_argument(
        '--gaze360label_file_train', dest='gaze360label_file_train', help='File path for gaze labels.',
        default='../gaze360_train/Label/train.label', type=str)      
    parser.add_argument(
        '--gaze360image_dir_val', dest='gaze360image_dir_val', help='Directory path for gaze images.',
        default='../gaze360_val/Image', type=str)
    parser.add_argument(
        '--gaze360label_dir_val', dest='gaze360label_dir_val', help='Directory path for gaze labels.',
        default='../gaze360_val/Label', type=str)
    
    # mpiigaze
    parser.add_argument(
        '--gazeMpiimage_dir', dest='gazeMpiimage_dir', help='Directory path for gaze images.',
        default='../MPII_norm/Image', type=str)
    parser.add_argument(
        '--gazeMpiilabel_dir', dest='gazeMpiilabel_dir', help='Directory path for gaze labels.',
        default='../MPII_norm/Label', type=str)


    parser.add_argument(
        '--output', dest='output', help='Path of output models.',
        default='output/snapshots/', type=str)
    parser.add_argument(
        '--snapshot', dest='snapshot', help='Path of model snapshot.',
        default='', type=str)
    parser.add_argument(
        '--gpu', dest='gpu_id', help='GPU device id to use [0] or multiple 0,1,2,3',
        default='0', type=str)

    ## MODEL ARGS
    
    parser.add_argument(
        '--num_epochs', dest='num_epochs', help='Maximum number of training epochs.',
        default=60, type=int)
    parser.add_argument(
        '--batch_size', dest='batch_size', help='Batch size.',
        default=1, type=int)
    parser.add_argument(
        '--angle', dest='angle', help='Angle filter',
        default=180, type=int)
    parser.add_argument(
        '--alpha', dest='alpha', help='Regression loss coefficient.',
        default=1, type=float)
    parser.add_argument(
        '--beta', dest='beta', help='Classification loss coefficient.',
        default=1, type=float)
    parser.add_argument(
        '--lr', dest='lr', help='Base learning rate.',
        default=0.00001, type=float)
    parser.add_argument(
        '--decay', dest='decay', help='Learning rate decay.',
        default=0.000001, type=float)
    parser.add_argument(
        '--reg_only', dest='reg_only', help='Only use regression loss.',
        default=False, type=bool)
    parser.add_argument(
        '--augment', dest='augment', help='If dataset should be augmented',
        default=False, type=bool)
    parser.add_argument(
        '--bins', dest='bins', help='Model.num_bins',
        default=181, type=int)

    parser.add_argument(
        '--best_bin', dest='best_bin', help='If best_bin loss',
        default=0, type=float)
    
    # ---------------------------------------------------------------------------------------------------------------------
    # Important args ------------------------------------------------------------------------------------------------------
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()
    cudnn.enabled = True
    num_epochs = args.num_epochs
    batch_size = args.batch_size
    gpu = select_device(args.gpu_id, batch_size=args.batch_size)
    alpha = args.alpha
    output=args.output    
    
    val_transform = transforms.Compose([
        # transforms.Resize(448),
        transforms.Resize(224),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])
    train_transform = transforms.Compose([
        # transforms.Resize(448),
        transforms.Resize(224),
        transforms.RandomApply(torch.nn.ModuleList([
            transforms.ColorJitter(brightness=0.25, contrast=0.25, saturation=0.25, hue=0.25)
        ]), p=0.5),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])
    
    # model = ML2CS180()
    model = VRI_GazeNet(num_bins=args.bins)
    model.cuda(gpu)
    bins = model.num_bins

    binwidth = model.binwidth

    print("BINWIDTH", binwidth)

    # TRAIN
    # folder = os.listdir(args.gaze360label_dir_train)
    # folder.sort()
    # testlabelpathombined = [os.path.join(args.gaze360label_dir_train, j) for j in folder] 

    if args.augment:
        gaze360=datasets.Gaze360(args.gaze360label_file_train, args.gaze360image_dir_train, train_transform, args.angle, binwidth)
        mpii = datasets.Mpiigaze(args.gazeMpiilabel_dir, args.gazeMpiimage_dir, train_transform, args.angle, binwidth)

    else:
        gaze360=datasets.Gaze360(args.gaze360label_file_train, args.gaze360image_dir_train, val_transform, args.angle, binwidth)
        mpii = datasets.Mpiigaze(args.gazeMpiilabel_dir, args.gazeMpiimage_dir, val_transform, args.angle, binwidth)

    
    dataset = ConcatDataset([gaze360, mpii])
    
    print('Loading data.')
    train_loader_gaze = DataLoader(
        dataset=dataset,
        batch_size=int(batch_size),
        shuffle=True,
        num_workers=0,
        pin_memory=True)

    # # VALIDATION
    folder = os.listdir(args.gaze360label_dir_val)
    folder.sort()
    testlabelpathombined = [os.path.join(args.gaze360label_dir_val, j) for j in folder]
    gaze_dataset_val=datasets.Gaze360(testlabelpathombined,args.gaze360image_dir_val, val_transform, 180, binwidth)
    # gaze_dataset_val=datasets.Gaze360(testlabelpathombined,args.gaze360image_dir_val, transformations, 180, binwidth)
    
    val_loader = torch.utils.data.DataLoader(
        dataset=gaze_dataset_val,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True)
    

    # torch.backends.cudnn.benchmark = True

    day = datetime.now().day
    month = datetime.now().month
    summary_name = f"VRI-{bins}-{month_name[month]}-{day}-LR{args.lr}-DEC{args.decay}-BATCH{batch_size}-augment-CROSSENTROPY-{not args.reg_only}-alpha-{args.alpha}-beta-{args.beta}"
    output=os.path.join(output, summary_name)
    if not os.path.exists(output):
        os.makedirs(output)
    
    criterion = nn.CrossEntropyLoss().cuda(gpu)
    reg_criterion = nn.MSELoss().cuda(gpu)
    # reg_criterion = nn.L1Loss().cuda(gpu)
    # softmax = nn.Softmax(dim=1).cuda(gpu)

    idx_tensor = [idx for idx in range(model.num_bins)]
    idx_tensor = torch.FloatTensor(idx_tensor).cuda(gpu)
    
    optimizer_gaze = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), args.lr, weight_decay=args.decay)

    avg_MAE_train=[]
    avg_MAE_val=[]

    best_val_loss = float("inf")
    best_val_epoch = None
    best_val_model = None
    
    best_train_loss = float("inf")
    best_train_epoch = None
    best_train_model = None

    all_models = []

    beta = args.beta

    for epoch in range(num_epochs):
        avg_error_train = 0.0
        total_train = 0

        iter_gaze = 0
        count = 0
        
        model.train()
        for i, (images_gaze, labels_gaze, cont_labels_gaze,name) in enumerate(train_loader_gaze):

            # total_train += cont_labels_gaze.size(0)
            total_train += 2 * cont_labels_gaze.size(0)
            images_gaze = Variable(images_gaze).cuda(gpu)
            
            # Binned labels
            label_yaw_gaze = Variable(labels_gaze[:, 0]).cuda(gpu)
            label_pitch_gaze = Variable(labels_gaze[:, 1]).cuda(gpu)
            # Continuous labels
            label_yaw_cont_gaze = Variable(cont_labels_gaze[:, 0]).cuda(gpu)
            label_pitch_cont_gaze = Variable(cont_labels_gaze[:, 1]).cuda(gpu)

            #Mirror
            mirror_image = images_gaze.detach().clone()
            for i in range(len(mirror_image)):
                mirror_image[i] = torchvision.transforms.functional.hflip(mirror_image[i])

            # print(mirror_image)
            mirror_yaw_bin = [(model.num_bins -1 - binned_yaw) for binned_yaw in label_yaw_gaze]
            mirror_pitch_bin = [int(binned_pitch) for binned_pitch in label_pitch_gaze]
            mirror_pitch_cont = [pitch for pitch in label_pitch_cont_gaze]
            mirror_yaw_cont = [-yaw for yaw in label_yaw_cont_gaze]

            mirror_image = Variable(mirror_image).cuda(gpu)
            mirror_yaw_bin = Variable(torch.tensor(mirror_yaw_bin)).cuda(gpu)
            mirror_pitch_bin = Variable(torch.tensor(mirror_pitch_bin)).cuda(gpu)
            mirror_pitch_cont = Variable(torch.Tensor(mirror_pitch_cont)).cuda(gpu)
            mirror_yaw_cont = Variable(torch.Tensor(mirror_yaw_cont)).cuda(gpu)

            ##CALCULATE ORIGINAL
            # images_gaze = augmentation_transform(images_gaze)
            yaw_predicted_ar, pitch_predicted_ar = model(images_gaze)

            loss_pitch_gaze = beta * criterion(pitch_predicted_ar, label_pitch_gaze)
            loss_yaw_gaze = beta * criterion(yaw_predicted_ar, label_yaw_gaze)

            if epoch > 10 and count <1:
                print("Cross entropy ", loss_yaw_gaze)



            with torch.no_grad():
                pitch_predicted_cpu = torch.sum(pitch_predicted_ar * idx_tensor, 1).cpu() * binwidth - 180
                yaw_predicted_cpu = torch.sum(yaw_predicted_ar * idx_tensor, 1).cpu() * binwidth - 180
                label_pitch_cpu = cont_labels_gaze[:,1].float()*np.pi/180
                label_pitch_cpu = label_pitch_cpu.cpu()
                label_yaw_cpu = cont_labels_gaze[:,0].float()*np.pi/180
                label_yaw_cpu = label_yaw_cpu.cpu()

            pitch_predicted = torch.sum(pitch_predicted_ar * idx_tensor, 1) * binwidth - 180
            yaw_predicted = torch.sum(yaw_predicted_ar * idx_tensor, 1) * binwidth - 180

            loss_reg_pitch = reg_criterion(pitch_predicted, label_pitch_cont_gaze)
            loss_reg_yaw = reg_criterion(yaw_predicted, label_yaw_cont_gaze)

            if args.best_bin:
                y_idx = []
                for y in yaw_predicted_ar:
                    y = list(y)
                    y_max = max(y)
                    y_idx.append(y.index(y_max))
                y_idx = torch.Tensor(y_idx).cuda(gpu)

                p_idx = []
                for p in pitch_predicted_ar:
                    p = list(p)
                    p_max = max(p)
                    p_idx.append(p.index(p_max))
                p_idx = torch.Tensor(p_idx).cuda(gpu)

                # y_idx = yaw.index(max(yaw))
                # p_idx = pitch.index(max(pitch))
                y = y_idx * binwidth - 180
                p = p_idx * binwidth - 180

                # y.cuda(gpu)
                # p.cuda(gpu)

                loss_reg_pitch += args.best_bin * reg_criterion(p, label_pitch_cont_gaze)
                loss_reg_yaw += args.best_bin * reg_criterion(y, label_yaw_cont_gaze)

            
            if epoch > 10 and count <1:
                print("L2 loss ", loss_reg_yaw)
                count += 1

            loss_pitch_gaze += alpha * loss_reg_pitch
            loss_yaw_gaze += alpha * loss_reg_yaw

            loss_seq = [loss_yaw_gaze, loss_pitch_gaze]
            grad_seq = [torch.tensor(1.0).cuda(gpu) for _ in range(len(loss_seq))]
            optimizer_gaze.zero_grad(set_to_none=True)
            torch.autograd.backward(loss_seq, grad_seq)
            optimizer_gaze.step()

            with torch.no_grad():
                pitch_predicted_cpu = pitch_predicted_cpu*np.pi/180
                yaw_predicted_cpu = yaw_predicted_cpu*np.pi/180 

                for p,y,pl,yl in zip(pitch_predicted_cpu,yaw_predicted_cpu,label_pitch_cpu,label_yaw_cpu):
                    avg_error_train += angular(gazeto3d([p,y]), gazeto3d([pl,yl]))


            ####### CALCULATE MIRROR
            # mirror_image = augmentation_transform(mirror_image)
            yaw_predicted, pitch_predicted = model(mirror_image)

            # Cross entropy loss
            loss_pitch_gaze = beta * criterion(pitch_predicted, mirror_pitch_bin)
            loss_yaw_gaze = beta * criterion(yaw_predicted, mirror_yaw_bin)

            with torch.no_grad():
                pitch_predicted_cpu = torch.sum(pitch_predicted * idx_tensor, 1).cpu() * binwidth - 180
                yaw_predicted_cpu = torch.sum(yaw_predicted * idx_tensor, 1).cpu() * binwidth - 180

                # label_pitch_cpu = cont_labels_gaze[:,1].float()*np.pi/180
                label_pitch_cpu = mirror_pitch_cont.float()*np.pi/180
                label_pitch_cpu = label_pitch_cpu.cpu()
                # label_yaw_cpu = cont_labels_gaze[:,0].float()*np.pi/180
                label_yaw_cpu = mirror_yaw_cont.float()*np.pi/180
                label_yaw_cpu = label_yaw_cpu.cpu()

            pitch_predicted = torch.sum(pitch_predicted * idx_tensor, 1) * binwidth - 180
            yaw_predicted = torch.sum(yaw_predicted * idx_tensor, 1) * binwidth - 180

            loss_reg_pitch = reg_criterion(pitch_predicted, mirror_pitch_cont)
            loss_reg_yaw = reg_criterion(yaw_predicted, mirror_yaw_cont)

            if args.best_bin:
                y_idx = []
                for y in yaw_predicted_ar:
                    y = list(y)
                    y_max = max(y)
                    y_idx.append(y.index(y_max))
                y_idx = torch.Tensor(y_idx).cuda(gpu)

                p_idx = []
                for p in pitch_predicted_ar:
                    p = list(p)
                    p_max = max(p)
                    p_idx.append(p.index(p_max))
                p_idx = torch.Tensor(p_idx).cuda(gpu)

                # y_idx = yaw.index(max(yaw))
                # p_idx = pitch.index(max(pitch))
                y = y_idx * binwidth - 180
                p = p_idx * binwidth - 180

                # y.cuda(gpu)
                # p.cuda(gpu)

                loss_reg_pitch += args.best_bin * reg_criterion(p, label_pitch_cont_gaze)
                loss_reg_yaw += args.best_bin * reg_criterion(y, label_yaw_cont_gaze)

            # Total loss
            loss_pitch_gaze += alpha * loss_reg_pitch
            loss_yaw_gaze += alpha * loss_reg_yaw

            loss_seq = [loss_yaw_gaze, loss_pitch_gaze]
            grad_seq = [torch.tensor(1.0).cuda(gpu) for _ in range(len(loss_seq))]
            optimizer_gaze.zero_grad(set_to_none=True)
            torch.autograd.backward(loss_seq, grad_seq)
            optimizer_gaze.step()

            with torch.no_grad():
                pitch_predicted_cpu = pitch_predicted_cpu*np.pi/180
                yaw_predicted_cpu = yaw_predicted_cpu*np.pi/180 

                for p,y,pl,yl in zip(pitch_predicted_cpu,yaw_predicted_cpu,label_pitch_cpu,label_yaw_cpu):
                    avg_error_train += angular(gazeto3d([p,y]), gazeto3d([pl,yl]))



        model.eval()
        with torch.no_grad():
            total = 0
            avg_error = .0        
            for j, (images, labels, cont_labels, name) in enumerate(val_loader):
                images = Variable(images).cuda(gpu)
                total += cont_labels.size(0)

                label_yaw = cont_labels[:,0].float()*np.pi/180
                label_pitch = cont_labels[:,1].float()*np.pi/180
                

                yaw_predicted, pitch_predicted = model(images)
    
                # mapping from binned (0 to 28) to angels (-180 to 180)  
                pitch_predicted = torch.sum(pitch_predicted * idx_tensor, 1).cpu() * binwidth - 180
                yaw_predicted = torch.sum(yaw_predicted * idx_tensor, 1).cpu() * binwidth - 180

                pitch_predicted = pitch_predicted*np.pi/180
                yaw_predicted = yaw_predicted*np.pi/180

                for p,y,pl,yl in zip(pitch_predicted,yaw_predicted,label_pitch,label_yaw):
                    avg_error += angular(gazeto3d([p,y]), gazeto3d([pl,yl]))
                
        val_loss = avg_error/total
                
        val_loss = (avg_error_val/total_val)
        train_loss = (avg_error_train/total_train)

        print(f"Epoch {epoch}: val:{val_loss} ; train:{train_loss}")

        avg_MAE_val.append(val_loss)
        avg_MAE_train.append(train_loss)

        torch.save(model.state_dict(), output +'/'+'epoch_' + str(epoch) + '.pkl')

        
        all_models.append((model.state_dict().copy(), val_loss, train_loss, epoch))

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_val_epoch = epoch
            best_val_state_dict = model.state_dict().copy()
        
        if train_loss < best_train_loss:
            best_train_loss = train_loss
            best_train_epoch = epoch
            best_train_state_dict = model.state_dict().copy()
        

    print(F'BEST EPOCH (VAL): {best_val_epoch}')
    print(F'BEST LOSS (VAL): {best_val_loss}')
    print(F'BEST EPOCH (train): {best_train_epoch}')
    print(F'BEST LOSS (train): {best_train_loss}')
    

    # print("Generating plot..")
    # epoch_list = list(range(num_epochs))
    
    # fig = plt.figure()        
    # plt.xlabel('Epoch')
    # plt.ylabel('MAE (degrees)')
    # plt.title('Mean angular error')
    # plt.plot(epoch_list, avg_MAE_train, color='b', label='train')
    # plt.plot(epoch_list, avg_MAE_val, color='g', label='validation')

    # plt.legend(loc="upper left")
    # plt.locator_params(axis='x', nbins=num_epochs//3)
    # fig.savefig(os.path.join(output,+"vri.png"), format='png')
    # plt.show()
