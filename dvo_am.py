import argparse
from path import Path

import torch
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
import models
from tqdm import tqdm

import torchvision.transforms as transforms
import flow_transforms
from imageio import imread, imwrite
import numpy as np
from util import flow2rgb

def main():
    input_transform = transforms.Compose(
        [
            flow_transforms.ArrayToTensor(),
            transforms.Normalize(mean=[0, 0, 0], std=[255, 255, 255]),
            transforms.Normalize(mean=[0.411, 0.432, 0.45], std=[1, 1, 1]),
        ]
    )

    img1_file = []
    img2_file = []
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    network_data = torch.load('flownets_EPE1.951.pth',map_location=torch.device('cpu'))
    print("=> using pre-trained model '{}'".format(network_data["arch"]))
    model = models.__dict__[network_data["arch"]](network_data).to(device)
    print(model)
    model.eval()

    exit()

    img1 = input_transform(imread(img1_file))
    img2 = input_transform(imread(img2_file))
    input_var = torch.cat([img1, img2]).unsqueeze(0)


if __name__ == "__main__":
    main()
