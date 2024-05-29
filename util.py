import os
import numpy as np
import shutil
import torch
import pickle
from torch.utils.data import Dataset
import torchvision.transforms as transforms
import flow_transforms
from imageio import imread, imwrite
import imageio

def save_checkpoint(state, is_best, save_path, filename="checkpoint.pth.tar"):
    torch.save(state, os.path.join(save_path, filename))
    if is_best:
        shutil.copyfile(
            os.path.join(save_path, filename),
            os.path.join(save_path, "model_best.pth.tar"),
        )


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __repr__(self):
        return "{:.3f} ({:.3f})".format(self.val, self.avg)


def flow2rgb(flow_map, max_value):
    flow_map_np = flow_map.detach().cpu().numpy()
    _, h, w = flow_map_np.shape
    flow_map_np[:, (flow_map_np[0] == 0) & (flow_map_np[1] == 0)] = float("nan")
    rgb_map = np.ones((3, h, w)).astype(np.float32)
    if max_value is not None:
        normalized_flow_map = flow_map_np / max_value
    else:
        normalized_flow_map = flow_map_np / (np.abs(flow_map_np).max())
    rgb_map[0] += normalized_flow_map[0]
    rgb_map[1] -= 0.5 * (normalized_flow_map[0] + normalized_flow_map[1])
    rgb_map[2] += normalized_flow_map[1]
    return rgb_map.clip(0, 1)

def read_gt_file_to_dict(file_path):
    result_dict = {}

    with open(file_path, 'r') as file:
        for line in file:
            # Skip comments and empty lines
            if line.startswith('#') or line.strip() == '':
                continue

            # Split the line into parts
            parts = line.strip().split()
            if len(parts) == 8:
                timestamp = parts[0]
                values = tuple(map(float, parts[1:]))  # Convert the rest to float and store in a tuple
                result_dict[float(timestamp)] = values

    return result_dict

def read_rgb_file_to_dict(file_path):
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
        gtDict = read_gt_file_to_dict(path + 'groundtruth.txt')
        gtDictSorted = sorted(gtDict)
        rgbDict = read_rgb_file_to_dict(path + 'rgb.txt')
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