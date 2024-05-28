import argparse
from path import Path
import os
from tqdm import tqdm
import pickle
import torchvision.transforms as transforms
from imageio import imread, imwrite
import imageio
import numpy as np
import time
import datetime

import torch
from torch.utils.data import Dataset
import torch.optim as optim
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
import torch.nn as nn
from torch.nn.init import kaiming_normal_, constant_
import torch.optim as optim

# local libraries
from util import flow2rgb, AverageMeter, save_checkpoint
import util
import models
import flow_transforms
from convlstm import ConvLSTM
from deepVO_wMemory import DvoAm_EncPlusTrack, DvoAm_Encoder
from deepVO_wRefining import DvoAm_EncTrackRefining

def lossFunction(yPredLocalList, yGtLocalList, yPredGlobalList, yGtGlobalList, k = 1): #TUM uses k = 1
    lossLocal = 0
    for i in range(len(yPredGlobalList)):
        yPredTrans = np.array(yPredLocalList[i][0:2], dtype='float32') # x,y,z
        yPredRot = np.array(yPredLocalList[i][3:6], dtype='float32') # 4 angles (quaternion)
        yGtTrans = np.array(yGtLocalList[i][0:2], dtype='float32') # x,y,z
        yGtRot = np.array(yGtLocalList[i][3:6], dtype='float32') # 4 angles (quaternion)
        lossLocal += np.inner(yPredTrans - yGtTrans) + k*np.inner(yPredRot - yGtRot)
    lossGlobal = 0
    for i in range(len(yPredGlobalList)):
        yPredTrans = np.array(yPredGlobalList[i][0:2], dtype='float32') # x,y,z
        yPredRot = np.array(yPredGlobalList[i][3:6], dtype='float32') # 4 angles (quaternion)
        yGtTrans = np.array(yGtGlobalList[i][0:2], dtype='float32') # x,y,z
        yGtRot = np.array(yGtGlobalList[i][3:6], dtype='float32') # 4 angles (quaternion)
        lossGlobal += np.inner(yPredTrans - yGtTrans) + k*np.inner(yPredRot - yGtRot)
    loss = lossLocal +  lossGlobal
    return loss


def lossFunctionLocal(yPredLocalList, yGtLocalList, k = 1): #TUM uses k = 1
    lossLocal = 0
    for batchIndex in range(yPredLocalList.shape[0]):
        traTensorErr = yPredLocalList[batchIndex][0:3] - yGtLocalList[batchIndex][0:3]
        rotTensorErr = yPredLocalList[batchIndex][3:7] - yGtLocalList[batchIndex][3:7]
        #print(traTensorErr)
        #print(rotTensorErr)
        traTensor = torch.dot(traTensorErr, traTensorErr)
        rotTensor = torch.dot(rotTensorErr, rotTensorErr)
        lossLocal += traTensor + k*rotTensor
    loss = lossLocal / yPredLocalList.shape[0] # average the loss of all batches
    return loss

def averagePoseGet(pose1, pose2, distance1, distance2):
    newPose = []
    for i in range(len(pose1)):
        # this is a weighted average between the two closest poses
        newPose.append((pose1[i]*distance2 + pose2[i]*distance1)/(distance1 + distance2))
        # this is just the closes timestamp: pose1
        #newPose.append(pose1[i])
    return tuple(newPose)

def datasetsListGet(paths):
    '''
    need to capture the images and the ground truth
    input: tuple(frame[k-1], frame[k])
    output: tuple(7d pose - xyz + 4 quaternions)'''
    datasetIO = []
    for path in paths:
        gtDict = util.read_gt_file_to_dict(path + 'groundtruth.txt')
        gtDictSorted = sorted(gtDict)
        rgbDict = util.read_rgb_file_to_dict(path + 'rgb.txt')
        rgbDictSorted = sorted(rgbDict)
        for i in range(len(gtDictSorted)-1):
            if(gtDictSorted[i+1] < gtDictSorted[i]):
                # if condition matched update the out
                print('error in order of gt')
        for i in range(len(rgbDictSorted)-1):
            if(rgbDictSorted[i+1] < rgbDictSorted[i]):
                # if condition matched update the out
                print('error in order of rgb')

        # gt has more elements than rgb, so we get the closest elements of rgb that are inside gt
        # Matching GT to RGB timestamps - we need to "fake" matching keypoints

        for i in range(1,len(rgbDictSorted)):
            frameCurrTimeStamp = rgbDictSorted[i]
            framePrevTimeStamp = rgbDictSorted[i-1]
            gtSortedRelativeTS = sorted(gtDictSorted, key=lambda x: abs(x - frameCurrTimeStamp))
            closest1, closest2 = gtSortedRelativeTS[:2]
            distance1 = abs(closest1 - frameCurrTimeStamp)
            distance2 = abs(closest2 - frameCurrTimeStamp)
            gtPose = averagePoseGet(gtDict[closest1], gtDict[closest2], distance1, distance2)
            input = (path + rgbDict[framePrevTimeStamp], path + rgbDict[frameCurrTimeStamp])
            datasetIO.append((input,gtPose))
        print('\n')
    return datasetIO



def trainModel(model, dataLoaderTrain, dataLoaderTest, device):
    # learning rate decays every 60k iterations
    optimizer = optim.Adam(model.parameters(), lr=0.0001, betas=(0.9, 0.99), weight_decay=0.0004)  # Adjust the LR as needed
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=60000, gamma=0.5)

    numIterations = 150000
    print(len(dataLoaderTrain))
    iterPerEpoch = len(dataLoaderTrain)
    numEpochs = numIterations // iterPerEpoch + 1  # Ensure we have enough epochs
    print("num of epochs: ", numEpochs)
    model.train()  # Set the model to training mode
    iteration = 0
    for epoch in range(numEpochs):

        train_loss = 0
        model.train()
        for images, poses in dataLoaderTrain:
            poses = poses.to(device)
            if iteration >= numIterations:
                break
            #images, poses = images.cuda(), poses.cuda()  # Move data to GPU if available

            # do i need this below?
            optimizer.zero_grad()  # Zero the parameter gradients
            #print('onias4')
            outputs = model(images)
            #print("outputs: ", outputs)
            #print("gt: ", poses)

            loss = lossFunctionLocal(outputs, poses)
            loss.backward()  # Backpropagation
            optimizer.step()  # Optimize the parameters
            train_loss += loss.item()
            if iteration % 100 == 0:
                print(f"Iteration {iteration}/{numIterations}, Loss: {loss.item()}")

            scheduler.step()  # Update the learning rate

            iteration += 1
        train_loss /= len(dataLoaderTrain.dataset)


        val_loss = 0
        model.eval()
        with torch.no_grad():
            for input, target in dataLoaderTest:
                target = target.to(device)
                output = model(input)
                loss = lossFunctionLocal(output, target)
                val_loss += loss.item()
        val_loss /= len(dataLoaderTest.dataset)
        if iteration >= numIterations:
            break
        print("Epoch:{} Training Loss:{:.2f} Validation Loss:{:.2f}\n".format(
            epoch, train_loss, val_loss))

    print("Training complete.")

def train_v2(train_loader, model, optimizer, train_writer, device):
    global n_iter, args
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    flow2_EPEs = AverageMeter()

    epoch_size = (
        len(train_loader)
        if args.epoch_size == 0
        else min(len(train_loader), args.epoch_size)
    )

    # switch to train mode
    model.train()

    end = time.time()

    for i, (input, target) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)
        target = target.to(device)
        input = torch.cat(input, 1).to(device)

        # compute output
        output = model(input)
        if args.sparse:
            # Since Target pooling is not very precise when sparse,
            # take the highest resolution prediction and upsample it instead of downsampling target
            h, w = target.size()[-2:]
            output = [F.interpolate(output[0], (h, w)), *output[1:]]


        loss = multiscaleEPE(
            output, target, weights=args.multiscale_weights, sparse=args.sparse
        )


        flow2_EPE = args.div_flow * realEPE(output[0], target, sparse=args.sparse)


        # record loss and EPE
        losses.update(loss.item(), target.size(0))
        train_writer.add_scalar("train_loss", loss.item(), n_iter)
        flow2_EPEs.update(flow2_EPE.item(), target.size(0))

        # compute gradient and do optimization step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            print(
                "Epoch: [{0}][{1}/{2}]\t Time {3}\t Data {4}\t Loss {5}\t EPE {6}".format(
                    epoch, i, epoch_size, batch_time, data_time, losses, flow2_EPEs
                )
            )
        n_iter += 1
        if i >= epoch_size:
            break

    return losses.avg, flow2_EPEs.avg


class CustomTUMDataset(Dataset):
    ''' datasetList is a list of tuples:
        The first element is a tuple of consecutive images
        The second element is the pose of the images.
    '''
    def __init__(self, data_list, device, transform=None):
        self.data_list = data_list
        self.transform = transform
        self.device    = device

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        input_transform = transforms.Compose(
            [
                flow_transforms.ArrayToTensor(),
                transforms.Normalize(mean=[0, 0, 0], std=[255, 255, 255]),
                transforms.Normalize(mean=[0.411, 0.432, 0.45], std=[1, 1, 1]),
            ]
        )

        item = self.data_list[idx]
        imagesPaths = item[0]
        pose = item[1]
        image0 = input_transform(imageio.v2.imread(imagesPaths[0])).to(self.device)
        image1 = input_transform(imageio.v2.imread(imagesPaths[1])).to(self.device)
        image = torch.cat([image0, image1])#.unsqueeze(0)

        pose = torch.tensor(pose, dtype=torch.float32)
        image.to(self.device)
        pose.to(self.device)
        return image, pose

def datasetsGet(trainPaths, testPaths):
    datasetsTrain = []
    if (os.path.isfile('trainDatasets.pkl') == False):
        datasetsTrain = datasetsListGet(trainPaths)
        with open('trainDatasets.pkl', 'wb') as file:
            pickle.dump(datasetsTrain, file)
    else:
        print("loading train pickle file")
        with open('trainDatasets.pkl', 'rb') as file:
            datasetsTrain = pickle.load(file)
    datasetsTest = []
    if (os.path.isfile('testDatasets.pkl') == False):
        datasetsTest = datasetsListGet(testPaths)
        with open('testDatasets.pkl', 'wb') as file:
            pickle.dump(datasetsTrain, file)
    else:
        print("loading test pickle file")
        with open('testDatasets.pkl', 'rb') as file:
            datasetsTest = pickle.load(file)

    return datasetsTrain, datasetsTest

def main():

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    print("device: ", device)

    #need to rerun the datasetsGet if i add more videos to the dataset
    trainPaths = ['datasets/rgbd_dataset_freiburg1_desk/',\
                  'datasets/rgbd_dataset_freiburg2_xyz/']
    testPaths = ['datasets/rgbd_dataset_freiburg2_desk/']

    datasetsTrain, datasetsTest = datasetsGet(trainPaths= trainPaths,
                                              testPaths=testPaths)
    #print(datasetsTrain)
    isTrain = True
    isTest = not isTrain

    tumDatasetTrain = CustomTUMDataset(datasetsTrain,device)
    tumDatasetTest = CustomTUMDataset(datasetsTest,device)
    train_loader = torch.utils.data.DataLoader(tumDatasetTrain, batch_size=4, shuffle=True)
    test_loader  = torch.utils.data.DataLoader(tumDatasetTest, batch_size=4, shuffle=True)
    #print(tumDataset.__getitem__(3))


    FlowNet_data = torch.load('flownets_EPE1.951.pth',map_location=torch.device('cpu'))
    print("=> using pre-trained model '{}'".format(FlowNet_data["arch"]))
    FlowNetModel = models.__dict__[FlowNet_data["arch"]](FlowNet_data).to(device)
    FlowNetModel.to(device)

    #model2 = DvoAm_EncPlusTrack(device=device, batchNorm=False).to(device)
    model2 = DvoAm_EncTrackRefining(device=device, batchNorm=False).to(device)

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

    print('onias1')
    new_state_dict = model2.state_dict()
    for name, param in subset_state_dict.items():
        #print(name)
        newName = 'encoding.'+name
        if newName in new_state_dict:
            new_state_dict[newName].copy_(param) #copy_() performs copy in place
    print(new_state_dict.keys())
    model2.load_state_dict(new_state_dict)
    print('onias2')
    print(datetime.datetime.now())
    if (isTrain):
        print("training")
        trainModel(model2, train_loader, test_loader, device)
        model_path = 'output_model.pth'
        torch.save(model2.state_dict(), model_path)
    else:
        print("testing")
    print(datetime.datetime.now())

#comecei a rodar as 16h33min

if __name__ == "__main__":
    main()
