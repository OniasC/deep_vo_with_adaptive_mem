import torch
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
import torch.nn as nn
from torch.nn.init import kaiming_normal_, constant_


import numpy as np

#from convlstm import ConvLSTM
#from convlstm2 import ConvLSTM
from convlstm_mine import ConvLSTM, ConvLSTMCell

def conv(batchNorm, in_planes, out_planes, kernel_size=3, stride=1):
    if batchNorm:
        return nn.Sequential(
            nn.Conv2d(
                in_planes,
                out_planes,
                kernel_size=kernel_size,
                stride=stride,
                padding=(kernel_size - 1) // 2,
                bias=False,
            ),
            nn.BatchNorm2d(out_planes),
            nn.LeakyReLU(0.1, inplace=True),
        )
    else:
        return nn.Sequential(
            nn.Conv2d(
                in_planes,
                out_planes,
                kernel_size=kernel_size,
                stride=stride,
                padding=(kernel_size - 1) // 2,
                bias=True,
            ),
            nn.LeakyReLU(0.1, inplace=True),
        )

class DvoAm_Full(nn.Module):
    expansion = 1

    def __init__(self, device, batchNorm=True):
        super(DvoAm_Full, self).__init__()

        self.device = device

        self.spatialAtten = False
        self.temporalAtten = True

        self.sizeMemory = 11

        self.memory = [] #memory should be a tensor of dimension (batch, N). Should it??

        self.thetaRot = 0.01 # unit: rad
        self.thetaTra = 0.01 # unit: m

        # Encoding
        self.batchNorm = batchNorm
        self.conv1 = conv(self.batchNorm, 6, 64, kernel_size=7, stride=2)
        self.conv2 = conv(self.batchNorm, 64, 128, kernel_size=5, stride=2)
        self.conv3 = conv(self.batchNorm, 128, 256, kernel_size=5, stride=2)
        self.conv3_1 = conv(self.batchNorm, 256, 256)
        self.conv4 = conv(self.batchNorm, 256, 512, stride=2)
        self.conv4_1 = conv(self.batchNorm, 512, 512)
        self.conv5 = conv(self.batchNorm, 512, 512, stride=2)
        self.conv5_1 = conv(self.batchNorm, 512, 512)
        self.conv6 = conv(self.batchNorm, 512, 1024, stride=2)

        # to say that i dont need to train these after initialzing
        for param in self.conv1[0].parameters(): #this covers weights and biases
            param.requires_grad = False
        for param in self.conv2[0].parameters():
            param.requires_grad = False
        for param in self.conv3[0].parameters():
            param.requires_grad = False
        for param in self.conv3_1[0].parameters():
            param.requires_grad = False
        for param in self.conv4[0].parameters():
            param.requires_grad = False
        for param in self.conv4_1[0].parameters():
            param.requires_grad = False
        for param in self.conv5[0].parameters():
            param.requires_grad = False
        for param in self.conv5_1[0].parameters():
            param.requires_grad = False
        for param in self.conv6[0].parameters():
            param.requires_grad = False

        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                kaiming_normal_(m.weight, 0.1)
                if m.bias is not None:
                    constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                constant_(m.weight, 1)
                constant_(m.bias, 0)

        # Tracking
        self.hiddenLocalLstm = torch.zeros((4,1024,8,10), device=self.device)
        self.localConvLSTM = ConvLSTMCell(in_channels=1024, out_channels=1024, kernel_size=(3,3), padding=(1,1), frame_size=(8,10), batch_size=4, device=device, activation="tanh") # what do these inputs mean??
        self.localGap = nn.AvgPool3d((1,8,10))
        # (1,1,1024)
        self.localFc = nn.Linear(in_features=1024,out_features=6) # 3 position + 3 angles

        #TODO: review these initializations
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                kaiming_normal_(m.weight, 0.1)
                if m.bias is not None:
                    constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                constant_(m.weight, 1)
                constant_(m.bias, 0)

        # entrada do tensor das duas imagens H x W x 6 (2 rgb's)

        # para entrada de 640 x 480 x 6, a saida eh de 10 x 8 x 1024
        # shape of tensor: [4, 1024, 8, 10]. 4 is the batch
        #output do encodingPlusTracking eh [4,6] 4 batches

        # convlstm2.py
        self.hiddenA = torch.zeros((4,1024,8,10), device=self.device)
        self.convLSTM = ConvLSTMCell(in_channels=1024, out_channels=1024, kernel_size=(3,3), padding=(1,1), frame_size=(8,10), batch_size=4, device=device, activation="tanh") # what do these inputs mean??

        self.refConv1 = conv(self.batchNorm, in_planes=2048, out_planes=1024, kernel_size=3, stride=1)
        self.refConv2 = conv(self.batchNorm, 1024, 1024, kernel_size=3, stride=1)

        # convnLTSM nao muda tamanho do tensor.. ou muda??
        self.gap = nn.AvgPool3d((1,8,10))
        # (1,1,1024)
        self.fc = nn.Linear(in_features=1024,out_features=6) # 3 position + 3 angles
        # a saida eh 6

        self.outA = torch.empty(4,1024,8,10).to(device)

        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                kaiming_normal_(m.weight, 0.1)
                if m.bias is not None:
                    constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                constant_(m.weight, 1)
                constant_(m.bias, 0)

    def forward(self, x):
        '''
        para o frame K:
        ---tracking---
        x_k = encoding(Frame_k, frame_{k-1})
        o_k, h_k = LSTM(x_k, h_{k-1})
        p_{k,k-1} = SE3(o_k)
        ---refining---
        check if you need to update M array
        if yes, update with latest h_k

        M'_k = f(oA_{k-1}, o_{k-1}, M_k), I know f. Need to double check if o_{k-1} is correct or is a spelling mistake
        x'_k = g1(oA_{k-1},M'_K), g1 is the same as function f !!! I get it! It isnt really. Its the same purpose, so maybe some internal product?

        tempTensor = stacked(x'_k,M'_K) #dim is now H x W x 2C
        xA_k = conv(tempTensor). paper says "2 conv layers of kernel size of 3 for fusion." So the dimensions are back to H x W x C.
                                ^ but what is the stride and what is the padding of the conv? Also, the paper appendix says its only one conv layer that doesnt lower dimension...
                                ^ I think i will first believe the paper, not the appendix.
        oA_k, hA_k = LSTM_A(xA_k, hA_{k-1})
        pA_k = SE3(oA_k)


        '''
        #encoding
        self.imgFlow = self.encoding(x)

        #tracking

        #transform a [4, 1024, 8, 10] tensor into a [4, 1024, 1, 8, 10]
        self.outLocalLstm, self.hiddenLocalLstm = self.localConvLSTM(self.imgFlow, self.hiddenLocalLstm)
        localPose = self.se3(self.outLocalLstm, self.localGap, self.localFc)
        #print("local pose: ", localPose)

        #refining

        newMemoryCandidate = (self.hiddenLocalLstm, localPose)
        #print(newHiddenState)

        self.memory = self.memoryUpdate(self.memory, newMemoryCandidate, x.size(0))

        M_dash = self.guideMemory(self.outA, self.hiddenLocalLstm, self.hiddenLocalLstm.shape) #outPrev or localPose??
        if (len(self.memory) == 0):
            self.outA = self.outLocalLstm
        x_dash = self.guideEncoding(self.imgFlow, self.outA)
        tempTensor = torch.cat((x_dash, M_dash), dim=1) # whatever is the dim of number of channels.
        xA = self.refConv2(self.refConv1(tempTensor))
        self.outA, self.hiddenA = self.convLSTM(xA, self.hiddenA)
        absPose = self.se3(self.outA, self.gap, self.fc)

        return absPose, localPose

    def encoding(self, input):
        out_conv1   = self.conv1(input)
        #print("out_conv1 ", out_conv1.shape)
        out_conv2   = self.conv2(out_conv1)
        #print("out_conv2 ", out_conv2.shape)
        out_conv3   = self.conv3(out_conv2)
        #print("out_conv3 ", out_conv3.shape)
        out_conv3_1 = self.conv3_1(out_conv3)
        #print("out_conv3_1 ", out_conv3_1.shape)
        out_conv4 = self.conv4(out_conv3_1)
        out_conv4_1 = self.conv4_1(out_conv4)
        #print("out_conv4_1 ", out_conv4_1.shape)
        out_conv5 = self.conv5(out_conv4_1)
        out_conv5_1 = self.conv5_1(out_conv5)
        #print("out_conv5_1 ", out_conv5_1.shape)
        out_conv6 = self.conv6(out_conv5_1)
        return out_conv6

    def se3(self, lstmOut, gap, fc):
        gapOut = gap(lstmOut)
        gapOut = torch.squeeze(gapOut, -1)
        gapOut = torch.squeeze(gapOut, -1)
        pose = fc(gapOut)
        return pose

    def memoryUpdate(self, memory: tuple, newMemoryCandidate: tuple, numBatches):
        '''
        break down and update memory differently for each batch
            memory is a tuple of hidden state (numBatches, C, H, W) and pose (4x7)

        '''
        newMemory = []
        isKeyFramePerBatch = []
        for iter in range(numBatches):
            isKeyFramePerBatch.append(False)

        if(len(memory) == 0):
            for iter in range(len(isKeyFramePerBatch)):
                isKeyFramePerBatch[iter] = True
        else:
            # content of memory are the (hidden states of key frames, pose)
            lastMemory = memory[-1]
            for i in range(len(isKeyFramePerBatch)):
                lastMemSingleBatch = lastMemory[i].unsqueeze(0)  # Extract a single batch, maintain 4D shape
                newMemSingleBatch = newMemoryCandidate[i].unsqueeze(0)  # Extract a single batch, maintain 4D shape
                isKeyFramePerBatch[i] = self.keyFrameCriteria(newMemSingleBatch, lastMemSingleBatch)

        if (len(self.memory) == self.sizeMemory):
            #just update the list,
            for index in range(len(self.memory)-1):
                self.memory[i][index] = self.memory[i][index + 1]
            self.memory[i][-1] = newMemoryCandidate[i]
        else:
            # size isnt max yet, so append to it
            self.memory.append(newMemoryCandidate)
        newMemory = self.memory
        return newMemory

    def guideMemory(self, outAbsPrev, outPrev, M_shape):
        '''
        M_dash = SUM(i: 1 to N)(alfa_i * m_dash_i)
        '''
        M_dash = torch.empty(M_shape).to(self.device)
        totalCosSim = torch.empty((M_shape[0],M_shape[2],M_shape[3])).to(self.device)
        for memory in self.memory:
            alfa_iNum = torch.exp(F.cosine_similarity(outAbsPrev, memory[0]))
            totalCosSim += alfa_iNum
        for i in range(len(self.memory)):
            alfa_i = torch.exp(F.cosine_similarity(outAbsPrev, self.memory[i][0]))/totalCosSim

            if (self.spatialAtten == True):
                beta = self.beta_iGet(outPrev, self.memory[i][0]) #NOT USING BETA FOR NOW..
                M_dash += alfa_i * self.comb(beta, self.memory[i][0])
            else:
                alfa_i = alfa_i.unsqueeze_(1)
                alfa_i = alfa_i.expand(-1,1024,-1,-1)
                M_dash += alfa_i * self.memory[i][0]
        return M_dash

    def beta_iGet(self, outPrev, memory):
        '''
        beta_j is the normalized weight between channel j of outPrev[j] and self.memory[i][j]. has size C
        beta needs to have the following dimensions:
                Batches x C x ??
                Batches: as anything else of this model
                C: because we will get the dot product og beta_j with m_j, where j goes from 1 to C
                ?? : what is the dimension of the cosine similarity
        beta calculation takes too long! will move to not have spatial atenuation (at least for now)
        '''
        # TODO: what should be the shape here??
        totalCosSim = torch.empty(4, 10).to(self.device) #total cossine similarity for a single channel
        beta = torch.empty(4, 1024, 10).to(self.device) # size of numBatches, 1024, H, W
        for j in range(memory.shape[1]):
            #for each channel:
            for k in range(outPrev.shape[1]):
                #print("outPrev ", outPrev[:,k,:,:])
                #print("memory ", memory[:,j,:,:])
                #print("cosine: ", torch.exp(F.cosine_similarity(outPrev[:,k,:,:], memory[:,j,:,:])))
                beta_jDen = torch.exp(F.cosine_similarity(outPrev[:,k,:,:], memory[:,j,:,:]))
                totalCosSim += beta_jDen
            # j goes from 0 to C-1 (1023) in our case
            beta[:,j,:] = torch.exp(F.cosine_similarity(outPrev[:,j,:,:], memory[:,j,:,:]))/totalCosSim
            print(j)
        return beta

    def guideEncoding(self, outEncoding, outAbsPrev):
        x_dash = outEncoding
        return x_dash

    def comb(beta, memory):
        '''
        C concatenates all reweighted feature maps along the channel dimension
        '''
        combination = torch.empty(beta.shape)
        for c in range(beta.shape[1]):
            combination[:,c,:,:] = beta[:,c,:,:]*memory[:,c,:,:]
        return combination

    def keyFrameCriteria(self, newHiddenState, lastHiddenState):
        newPose = newHiddenState[1]
        lastPose = lastHiddenState[1]

        isRotKey = False
        newAngle = newPose[3:6]
        prevAngle = lastPose[3:6]

        rotDiff = newAngle - prevAngle
        rotTensor = torch.dot(rotDiff, rotDiff)
        if (rotTensor > self.thetaRot):
            isRotKey = True

        isTraKey = False
        newPos = newPose[0:3]
        prevPos = lastPose[0:3]
        traDiff = newPos - prevPos
        rotTensor = torch.dot(traDiff, traDiff)
        if (traDiff > self.thetaTra):
            isTraKey = True

        return (isRotKey and isTraKey)