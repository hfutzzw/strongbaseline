# encoding: utf-8
import glob
import re
import os
import os.path as osp

from .bases import BaseImageDataset
import json

import numpy as np
import pdb

def read_txt(path,dir):
    """
    :param path: txt file path
    :param dir:
    :return:
    """
    # a fake camid
    camid = 0
    train_path = []

    with open(path,'r') as f:
        lines = f.readlines()
        for line in lines:
            img_path = osp.join(dir,osp.basename(line.strip().split()[0]))
            identity = int(line.strip().split()[1])
            train_path.append((img_path,identity,camid))

    return train_path

class NAICdataset(BaseImageDataset):
    dataset_dir = 'NAIC'

    def __init__(self, root='/home/hzr/bag_of_tricks/reid-strong-baseline/data/', verbose=True, **kwargs):
        super(NAICdataset, self).__init__()
        self.root = root
        self.dataset_dir = osp.join(root, self.dataset_dir)
        self.train_dir = osp.join(self.dataset_dir, 'NAICtrainset')
        self.test_dir = osp.join(self.dataset_dir,'NAICtestset')

        self.val_split_pid_number = 2

        # use validation testset from trainset
        use_split_testset = False

        train_set, test_query_set,test_gallery_set = \
            self.read_annotations_train(osp.join(self.train_dir, 'train_list.txt'),self.val_split_pid_number)

        if not use_split_testset:
            test_query_set = self.read_annotations_test_q(osp.join(self.test_dir,'query_b'))
            test_gallery_set = self.read_annotations_test_g(osp.join(self.test_dir, 'gallery_b'))

        print("=> NAICdataset loaded")
        self.print_dataset_statistics(train_set, test_query_set, test_gallery_set)

        self.train = train_set
        self.query = test_query_set
        self.gallery = test_gallery_set

        self.num_train_pids, self.num_train_imgs, self.num_train_cams = self.get_imagedata_info(self.train)
        self.num_query_pids, self.num_query_imgs, self.num_query_cams = self.get_imagedata_info(self.query)
        self.num_gallery_pids, self.num_gallery_imgs, self.num_gallery_cams = self.get_imagedata_info(self.gallery)


    def read_annotations_train(self,path,split_num):
        total_path = read_txt(path,dir=osp.join(self.train_dir,'train_set'))
        test_q = []
        test_g = []
        index = 0
        for i,ele in enumerate(total_path):

            if ele[1] == index :
                index = index + 1
                if index > split_num:
                    break
                test_q.append(ele)

            else:
                test_g.append(ele)

        train = total_path[i:]
        train = [ (t[0], t[1]-split_num, t[2]) for t in train]

        return train, test_q, test_g

    def read_annotations_test_q(self,dir):
        #return read_txt(path,dir= osp.join(self.test_dir,'query_a'))
        pid = 0
        camid = 0
        imgpath_list = [ osp.join(dir,img_name) for img_name in os.listdir(dir) ]
        test_g = [(img_path,pid,camid) for img_path in imgpath_list]
        return test_g


    def read_annotations_test_g(self,dir):
        pid = 0
        camid = 0
        imgpath_list = [ osp.join(dir,img_name) for img_name in os.listdir(dir) ]
        test_g = [(img_path,pid,camid) for img_path in imgpath_list]
        return test_g



















