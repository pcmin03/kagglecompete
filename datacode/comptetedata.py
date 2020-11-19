import numpy as np
import skimage ,numbers, random
from glob import glob
from natsort import natsorted
from tqdm import tqdm
from torch.utils.data import Dataset
import torch
from skimage.filters import threshold_otsu,threshold_yen , threshold_local
# from torchvision import transforms

from skimage.io import imsave

from skimage.util.shape import view_as_blocks,view_as_windows
import cv2

import torch.nn.functional as F
# set image smoothing
from albumentations.pytorch import transforms  
from albumentations import *
import torchvision
import albumentations

from scipy.ndimage import gaussian_filter
from skimage.morphology import skeletonize, thin
from skimage.morphology import erosion, dilation, opening, closing,disk

from sklearn.preprocessing import MultiLabelBinarizer, OneHotEncoder

import datacode.custom_transforms as custom_transforms
import fastremap

# import imgaug.augmenters as iaa
# aug = iaa.CoarseDropout((0.0, 0.05), size_percent=(0.02, 0.25))

# import cupy as cp
class mydataset_2d(Dataset):
    def __init__(self,imageDir,labelDir,patchsize,phase):
        # self.Ltransforms = albumentations.Compose([
        #             # albumentations.pytorch.transforms.ToTensorV2(),
        #             albumentations.Normalize((0.31789994),(0.19416974))])
        
        self.Ltransforms =  torchvision.transforms.Compose([
                        torchvision.transforms.ToTensor(),
                        torchvision.transforms.Normalize((0.31789994),(0.19416974))])

        self.T_transform = albumentations.RandomCrop(256, 256)
        self.centercrop = albumentations.CenterCrop(768,1024 )
        self.images = imageDir
        self.labels = labelDir
        self.phase = phase



    def __len__(self):
        return len(self.images)
    

    def __getitem__(self,index):

        image = skimage.io.imread(self.images[index])
        mask = skimage.io.imread(self.labels[index])
        
        mappings = {
        # 1 :18 ,2 :17 ,3 :16 ,4 :15 ,5 :14 ,6 :13 ,7 :12 ,8 :11,
        # 9 :21 ,10:22 ,11:23 ,12:24 ,13:25 ,14:26 ,15:27 ,16:28,
        # 17:38 ,18:37 ,19:36 ,20:35 ,21:34 ,22:33 ,23:32 ,24:31,
        # 25:41 ,26:42 ,27:43 ,28:44 ,29:45 ,30:46 ,31:47 ,32:48,
        # 33:255,34:254,35:127:0:255}
        18:1 , 17:2 , 16:3 , 15:4 , 14:5 , 13:6 , 12:7 , 11:8 ,
        21:9 , 22:10, 23:11, 24:12, 25:13, 26:14, 27:15, 28:16,
        38:17, 37:18, 36:19, 35:20, 34:21, 33:22, 32:23, 31:24,
        41:25, 42:26, 43:27, 44:28, 45:29, 46:30, 47:31, 48:32,
        255:33, 254:34, 127:35,253:36}

        mask = fastremap.remap(mask, mappings, preserve_missing_labels=True)
        if self.phase == 'train':    
            sample = self.T_transform(image=image, mask=mask)
            image = sample['image']
            mask = sample['mask']
        elif self.phase == 'valid':
            sample = self.centercrop(image=image, mask=mask)
            image = sample['image']
            mask = sample['mask']
        # print(iamge.shape)
            
        # A[i] -= torch.min(A[i])
        # A[i] /= torch.max(A[i])
        

        # if self.phase =='train':
        image = self.Ltransforms(image)
        
        return image, mask

