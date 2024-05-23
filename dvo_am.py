import argparse
from path import Path
import os
import torch
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
import models
from tqdm import tqdm

import torchvision.transforms as transforms
from imageio import imread, imwrite
import numpy as np
from util import flow2rgb

import torch.nn as nn
from torch.nn.init import kaiming_normal_, constant_

from convlstm import ConvLSTM

from deepVO_wMemory import DvoAm_EncPlusTrack, DvoAm_Encoder

import torch.optim as optim

def lossFunction(yPred, yGt):
    lossLocal = 1
    lossGlobal = 2
    loss = lossLocal +  lossGlobal
    return loss

def read_file_to_dict(file_path):
    result_dict = {}

    with open(file_path, 'r') as file:
        for line in file:
            # Skip comments and empty lines
            if line.startswith('#') or line.strip() == '':
                continue

            # Split the line into timestamp and filename
            parts = line.strip().split()
            if len(parts) == 2:
                timestamp, filename = parts
                result_dict[float(timestamp)] = filename

    return result_dict

#need to capture the images and the ground truth
def datasetsTrainGet():
    paths = ['datasets/rgbd_dataset_freiburg1_desk/',\
             'datasets/rgbd_dataset_freiburg2_xyz/']
    files = []
    filesDict = {}
    pathName = ""

    datasetIO = {}
    for path in paths:
        rgbPath = path+'rgb/'
        #open rgb dir
        frameTimeStamps = []
        for fileFullPath in Path(rgbPath).files():
            fileName = os.path.basename(fileFullPath)
            pathName = os.path.dirname(fileFullPath)
            fileName = fileName.rsplit(".",1)[0]
            frameTimeStamps.append(float(fileName))
            filesDict[float(fileName)] = fileName
        frameTimeStamps.sort()


        datasetIO[path] = ()
    print(files)
    exit()
    test_files = []
    count = 1
    for file in files:
        fullPathSource = pathName + '/' + filesDict[file] + '.' + ext
        fullPathDest   = pathName + '/' + str(count) + '.' + ext
        if (os.path.exists(fullPathDest) == False):
            os.popen('cp ' + fullPathSource + ' ' + fullPathDest)
        test_files.append(os.path.normpath(fullPathDest))
        count += 1
    #print("\n\n\tAAAAAAAAAAAAAAAAAA\n\n")
    print(test_files)


def datasetsTestGet():
    paths = ['datasets/rgbd_dataset_freiburg2_desk/']

def train(model):
    # learning rate decays every 60k iterations
    learningRateInitial = 0.0001
    optimizer = optim.Adam(model.parameters(), lr=learningRateInitial, betas=(0.9, 0.99), weight_decay=0.0004)  # Adjust the LR as needed

    numEpochs = 10
    batchSize = 4

    for epoch in range(numEpochs):
        for i in range(0, len(X), batchSize):
            batchX = 3#....
            batchY = 3#....

            optimizer.zero_grad()
            outputs = model(batchX)
            loss = lossFunction()
            loss.backwards()
            optimizer.step()
        print(f"Epoch [{epoch+1}/{numEpochs}], Loss: {loss.item():.4f}")

def main():
    datasetsTrainGet()
    exit()
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    FlowNet_data = torch.load('flownets_EPE1.951.pth',map_location=torch.device('cpu'))
    print("=> using pre-trained model '{}'".format(FlowNet_data["arch"]))
    FlowNetModel = models.__dict__[FlowNet_data["arch"]](FlowNet_data).to(device)

    model2 = DvoAm_EncPlusTrack(batchNorm=False)
    #Capturing the layers we want from FlowNet!
    encodingLayers = ['conv1.0.weight', 'conv2.0.weight', \
                    'conv3.0.weight', 'conv3_1.0.weight', \
                    'conv4.0.weight', 'conv4_1.0.weight', \
                    'conv5.0.weight', 'conv5_1.0.weight', \
                    'conv6.0.weight',\
                    'conv1.0.bias', 'conv2.0.bias', \
                    'conv3.0.bias', 'conv3_1.0.bias', \
                    'conv4.0.bias', 'conv4_1.0.bias', \
                    'conv5.0.bias', 'conv5_1.0.bias', \
                    'conv6.0.bias' ]
    subset_state_dict = {}
    for name, param in FlowNetModel.named_parameters():
        if name in encodingLayers:  # Change 'desired_layer' to the relevant layer names
            subset_state_dict[name] = param
    #print(subset_state_dict)


    new_state_dict = model2.state_dict()
    for name, param in subset_state_dict.items():
        #print(name)
        newName = 'encoding.'+name
        if newName in new_state_dict:
            new_state_dict[newName].copy_(param) #copy_() performs copy in place
    print(new_state_dict.keys())
    model2.load_state_dict(new_state_dict)

    exit()

    input_transform = transforms.Compose(
        [
            flow_transforms.ArrayToTensor(),
            transforms.Normalize(mean=[0, 0, 0], std=[255, 255, 255]),
            transforms.Normalize(mean=[0.411, 0.432, 0.45], std=[1, 1, 1]),
        ]
    )



    img1 = input_transform(imread(img1_file))
    img2 = input_transform(imread(img2_file))
    input_var = torch.cat([img1, img2]).unsqueeze(0)


if __name__ == "__main__":
    main()
