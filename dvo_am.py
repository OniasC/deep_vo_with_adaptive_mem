import argparse
from path import Path
import os
from tqdm import tqdm
import numpy as np
import time
import datetime

import torch
import torch.optim as optim
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
import torch.nn as nn
from torch.nn.init import kaiming_normal_, constant_
import torch.optim as optim

# local libraries
from util import flow2rgb, AverageMeter, save_checkpoint, datasetsGet, CustomTUMDataset
import util
import models

from convlstm import ConvLSTM
from deepVO_wMemory import DvoAm_EncPlusTrack, DvoAm_Encoder
from deepVO_wRefining import DvoAm_EncTrackRefining
from deep_vo_full import DvoAm_Full


def lossFunction(absPredList, localPredList, gt, onlyAbs = False, k = 1): #TUM uses k = 1
    '''
    L_local  = (1/t)\sum_{i from 1 to t}(||pos_rel - pos_rel_gt|| + k*||ang_rel - ang_rel_gt||)
    L_global = \sum_{i from 1 to t}(1/i)(||pos_abs - pos_abs_gt|| + k*||ang_abs - ang_abs_gt||)
    L_total = L_local + L_global (for training)
    L_total = L_global (for testing. "During the testing process, only the refined absolute poses are used as
                                        final results")
    '''
    lossPerBatch = []
    for i in range(len(absPredList)):

    for batchIndex in range(absPredList.shape[0]):
        absPred   = absPredList[batchIndex]
        localPred = localPredList[batchIndex]
        absGt   = gt[batchIndex][0:6]
        localGt   = gt[batchIndex][6:-1]

        localTraErr = localPred[0:3] - localGt[0:3]
        localRotErr = localPred[3:6] - localGt[3:6]
        absTraErr = absPred[0:3] - absGt[0:3]
        absRotErr = absPred[3:6] - absGt[3:6]

        tra = np.sqrt(torch.dot(traTensorErr, traTensorErr))
        rotTensor = np.sqrt(torch.dot(rotTensorErr, rotTensorErr))
        lossTotal = absTraErr + k*absRotTensor
        if (onlyAbs == True):
            lossTotal = 0
    return loss


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

        for sequence in dataLoaderTrain:
            framesSeq = []
            gtsSeq = []
            for frames, gt in sequence:
                framesSeq.append(frames)
                gtsSeq.append(gt.to(device))

            if iteration >= numIterations:
                break
            #images, poses = images.cuda(), poses.cuda()  # Move data to GPU if available

            # do i need this below?
            optimizer.zero_grad()  # Zero the parameter gradients
            #print('onias4')
            absPoseList, localPoseList = model(framesSeq)
            #print("outputs: ", outputs)
            #print("gt: ", poses)

            individualLoss = lossFunction(absPoseList, localPoseList, gtsSeq)
            batchLoss = individualLoss.mean()
            batchLoss.backward()  # Backpropagation
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
                absPose, localPose = model(input)
                loss = lossFunction(absPose, localPose, target)
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



def main():

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    print("device: ", device)

    #need to rerun the datasetsGet if i add more videos to the dataset
    trainPaths = ['datasets/rgbd_dataset_freiburg1_desk/',\
                  'datasets/rgbd_dataset_freiburg2_xyz/']
    testPaths = ['datasets/rgbd_dataset_freiburg2_desk/']
    opticalFlowModelFile = 'flownets_EPE1.951.pth'

    datasetsTrain, datasetsTest = datasetsGet(trainPaths= trainPaths,
                                              testPaths=testPaths)
    #print(datasetsTrain)
    isTrain = True
    isTest = not isTrain

    tumDatasetTrain = CustomTUMDataset(datasetsTrain,device)
    tumDatasetTest = CustomTUMDataset(datasetsTest,device)
    train_loader = torch.utils.data.DataLoader(tumDatasetTrain, batch_size=4, shuffle=True)
    test_loader  = torch.utils.data.DataLoader(tumDatasetTest, batch_size=4, shuffle=True)

    if (os.path.isfile(opticalFlowModelFile) == False):
        opticalFlowModelFile = opticalFlowModelFile + '.tar'

    FlowNet_data = torch.load(opticalFlowModelFile, map_location=torch.device(device))
    print("=> using pre-trained model '{}'".format(FlowNet_data["arch"]))
    FlowNetModel = models.__dict__[FlowNet_data["arch"]](FlowNet_data).to(device)
    FlowNetModel.to(device)

    #model2 = DvoAm_EncPlusTrack(device=device, batchNorm=False).to(device)
    model2 = DvoAm_Full(device=device, batchNorm=False).to(device)

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
