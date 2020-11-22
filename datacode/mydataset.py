import numpy as np
import skimage ,numbers , random
from glob import glob
from natsort import natsorted
import torch

from torchvision import transforms

import torch.nn.functional as F
# set image smoothing
from scipy.ndimage import gaussian_filter
from sklearn.model_selection import KFold

from numpy.lib.stride_tricks import as_strided as ast
from torch.utils.data import DataLoader

def resetseed(random_seed):
    torch.manual_seed(random_seed)
    np.random.seed(random_seed)
    random.seed(random_seed)
    torch.cuda.manual_seed(random_seed)
#  kfold data
def divide_kfold(Datadir,args):
    imageDir,labelDir,testDir,tlabelDir = Datadir
    images = np.array(natsorted(glob(imageDir+'*')))
    labels = np.array(natsorted(glob(labelDir+'*')))
    train,valid = dict(),dict()
    
    if args.cross_validation == True:
        # total_label = []
        # for label in labels: 
        #     total_label.append(np.array(natsorted(glob(label+'/*'))))
        # labels = np.array(total_label)
        kfold = KFold(n_splits=args.Kfold)
        i = 0
        
        # print(f"train_index{train_index} \t test_index:{test_index}")
        for train_index, test_index in kfold.split(images):
            img_train,img_test = images[train_index], images[test_index]
            lab_train,lab_test = labels[train_index], labels[test_index]
            i+=1
            
            train.update([('train'+str(i),img_train),('test'+str(i),img_test)])
            valid.update([('train'+str(i),lab_train),('test'+str(i),lab_test)])
        
        train_num, test_num = 'train'+str(args.knum), 'test'+str(args.knum)
        #train set
        image_train = train[train_num]
        label_train = valid[train_num]
        #valid set
        image_valid = train[test_num]
        label_valid = valid[test_num]
        
    else: 
        image_valid = np.array(natsorted(glob(testDir+'*')))
        label_valid = np.array(natsorted(glob(tlabelDir+'*')))
    # print([image_train,image_valid],[label_train,label_valid])
    return [image_train,image_valid],[label_train,label_valid]

def select_data(args):

    # if args.datatype == 'uint8':
        #uint8 train
    imageDir= '/workspace/hjjang/201113_2020healthhub_challenge/tif_img_v3_clipthr05/'
    labelDir = '/workspace/hjjang/201113_2020healthhub_challenge/mask_img_v3_multilabel_center_circle_rearr/'
    #uint8 test
    testDir= '/workspace/hjjang/201113_2020healthhub_challenge/tif_img_v3_clipthr05/'
    tlabelDir = '/workspace/hjjang/201113_2020healthhub_challenge/mask_img_v3_multilabel_center_circle_rearr/'
        # testDir ='../test_image/'
        # tlabelDir = '../test_label/'
    
    return [imageDir,labelDir,testDir,tlabelDir]

def make_dataset(trainset,validset,args): 
    num_workers = 16
    print(len(trainset[0]),len(trainset[1]),'len dataset')
    from .comptetedata import mydataset_2d
    MyDataset = {'train': DataLoader(mydataset_2d(trainset[0],validset[0],args.patchsize,phase='train'),
                    args.batch_size, 
                    shuffle = True,
                    num_workers = num_workers),
                'valid' : DataLoader(mydataset_2d(trainset[1],validset[1],args.patchsize,phase='valid'),
                        1, 
                        shuffle = False,
                        num_workers = num_workers)}

    return MyDataset