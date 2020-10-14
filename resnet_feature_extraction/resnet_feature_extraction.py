# -*- coding: utf-8 -*-
"""
Transfer Learning tutorial
==========================
**Author**: `Sasank Chilamkurthy <https://chsasank.github.io>`_

In this tutorial, you will learn how to train your network using
transfer learning. You can read more about the transfer learning at `cs231n
notes <http://cs231n.github.io/transfer-learning/>`__

Quoting these notes,

    In practice, very few people train an entire Convolutional Network
    from scratch (with random initialization), because it is relatively
    rare to have a dataset of sufficient size. Instead, it is common to
    pretrain a ConvNet on a very large dataset (e.g. ImageNet, which
    contains 1.2 million images with 1000 categories), and then use the
    ConvNet either as an initialization or a fixed feature extractor for
    the task of interest.

These two major transfer learning scenarios look as follows:

-  **Finetuning the convnet**: Instead of random initializaion, we
   initialize the network with a pretrained network, like the one that is
   trained on imagenet 1000 dataset. Rest of the training looks as
   usual.
-  **ConvNet as fixed feature extractor**: Here, we will freeze the weights
   for all of the network except that of the final fully connected
   layer. This last fully connected layer is replaced with a new one
   with random weights and only this layer is trained.

"""
# License: BSD
# Author: Sasank Chilamkurthy

from __future__ import print_function, division

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
import time
import os
import copy
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
import pandas as pd
from skimage import io, transform
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES=True
import scipy
import scipy.io
import pdb
plt.ion()   # interactive mode

# for reproducibility
import random
random.seed(0)
np.random.seed(0)
torch.manual_seed(0)
if torch.cuda.is_available():
    torch.cuda.manual_seed(0)
torch.backends.cudnn.deterministic = True
######################################################################
# Load Data
# ---------
#
# We will use torchvision and torch.utils.data packages for loading the
# data.
#
# The problem we're going to solve today is to train a model to classify
# **ants** and **bees**. We have about 120 training images each for ants and bees.
# There are 75 validation images for each class. Usually, this is a very
# small dataset to generalize upon, if trained from scratch. Since we
# are using transfer learning, we should be able to generalize reasonably
# well.
#
# This dataset is a very small subset of imagenet.
#
# .. Note ::
#    Download the data from
#    `here <https://download.pytorch.org/tutorial/hymenoptera_data.zip>`_
#    and extract it to the current directory.

# Data augmentation and normalization for training
# Just normalization for validation


data_transform = transforms.Compose([
        transforms.Resize((224,224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
work_dir = 'E:\\DomainAdaptation\\cifar-stl\\'
data_dir = 'E:\\DomainAdaptation\\cifar-stl\\stl\\test'
#work_dir = '/mnt/HD2T/DomainAdaptation/image-clef/'
#data_dir = '/mnt/HD2T/DomainAdaptation/image-clef/p/'
#work_dir = '/mnt/HD2T/DomainAdaptation/Office31data/'
#data_dir = '/mnt/HD2T/DomainAdaptation/Office31data/webcam/images'
#work_dir = '/mnt/HD2T/DomainAdaptation/OfficeHomeDataset_10072016/'
#data_dir = '/mnt/HD2T/DomainAdaptation/OfficeHomeDataset_10072016/Product'
#work_dir = '/mnt/HD2T/X-ray-Classification/datasetA-20classes/'
#data_dir = '/mnt/HD2T/X-ray-Classification/datasetA-20classes/regular-images'                
                
image_dataset = datasets.ImageFolder(data_dir,transform=data_transform)

## split the data into train/validation/test subsets
indices = list(range(len(image_dataset)))
dataloader_for_feature_extraction = torch.utils.data.DataLoader(dataset=image_dataset,batch_size=50,num_workers=0)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
######################################################################
# Feature extractor
#
def extract_features(model):

    model.eval()
    num_images = len(image_dataset)
    count = 0
    for index, (inputs, labels) in enumerate(dataloader_for_feature_extraction):
        print('Features extracted for {} out of {} images'.format(count,num_images))
        #inputs = inputs.to(device)
        #labels = labels.to(device)
        inputs = inputs.float()
        outputs = model(inputs).view(-1,2048)
        with torch.set_grad_enabled(False):
            outputs = model(inputs)
            
            # backward + optimize only if in training phase
            if index == 0:
                allFeatures = outputs
                allLabels = labels
            else:
                allFeatures = torch.cat((allFeatures, outputs), 0)
                allLabels = torch.cat((allLabels, labels), 0)
        count = count + outputs.shape[0]
    return allFeatures,allLabels

######################################################################
# ----------------------
#
# Load a pretrained model and reset final fully connected layer.
#
from myResnet import resnet101_feature_extractor,resnet50_feature_extractor
model_ft = resnet50_feature_extractor(pretrained=True)
#model_ft.load_state_dict(torch.load(work_dir+'/xray-image-classifier-resnet101.pt'),strict=False)
allFeatures,allLabels = extract_features(model_ft)
scipy.io.savemat(work_dir+'/cifarStl-stl-resnet50-noft.mat', mdict={'resnet50_features': allFeatures.to(torch.device("cpu")).numpy(), 'labels':allLabels.to(torch.device("cpu")).numpy()})
#scipy.io.savemat(work_dir+'/XrayDataset-regu-resnet101-noft.mat', mdict={'resnet101_features': allFeatures.to(torch.device("cpu")).numpy(), 'labels':allLabels.to(torch.device("cpu")).numpy()})
