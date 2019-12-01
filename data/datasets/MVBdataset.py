# encoding: utf-8
import glob
import re

import os.path as osp

from .bases import BaseImageDataset
import json

import numpy as np
import pdb


class MVBdataset(BaseImageDataset):
    dataset_dir = 'MVB_dataset'

    def __init__(self, root='/home/hzr/bag_of_tricks/reid-strong-baseline/data/', verbose=True, **kwargs):
        super(MVBdataset, self).__init__()
        self.root = root
        self.dataset_dir = osp.join(root, self.dataset_dir)
        self.train_dir = osp.join(self.dataset_dir, 'MVB_train')
        self.query_dir = osp.join(self.dataset_dir, 'MVB_mini')
        self.val_split_number = 4

        # self.gallery_dir = osp.join(self.dataset_dir, 'bounding_box_test')
        use_split_testset = False

        train_set, test_gallery_set, test_query_set = \
            self.read_annotations_train(osp.join(root, 'MVB_dataset/MVB_train/Info/train.json'))

        if not use_split_testset:
            test_query_set = self.read_annotations_test_q(osp.join(root, 'MVB_dataset/MVB_val/Info/val_probe.json'))
            test_gallery_set = self.read_annotations_test_g(osp.join(root, 'MVB_dataset/MVB_val/Info/val_gallery.json'))

        print("=> MVBdataset loaded")
        self.print_dataset_statistics(train_set, test_query_set, test_gallery_set)

        self.train = train_set
        self.query = test_query_set
        self.gallery = test_gallery_set

        self.num_train_pids, self.num_train_imgs, self.num_train_cams = self.get_imagedata_info(self.train)
        self.num_query_pids, self.num_query_imgs, self.num_query_cams = self.get_imagedata_info(self.query)
        self.num_gallery_pids, self.num_gallery_imgs, self.num_gallery_cams = self.get_imagedata_info(self.gallery)

    def read_annotations_train(self, dir):
        with open(dir, 'r') as input_json:
            train_annotation = json.load(input_json)

        test_annotation_split = train_annotation['image'][:self.val_split_number]
        train_annotation_split = train_annotation['image'][self.val_split_number:]

        query_ids = []

        included_anno_query, included_anno_gallery, query_ids = self.build_testingset_by_spliting_traingset(
            test_annotation_split)

        # included_anno_query = self.read_annotations_test_q(osp.join(self.root,'MVB_dataset/MVB_val/Info/val_probe.json'))
        # included_anno_gallery = self.read_annotations_test_g(osp.join(self.root,'MVB_dataset/MVB_val/Info/val_gallery.json'))

        # pdb.set_trace()

        included_anno_train = []

        materials_dict = {'paperboard': 0,
                          'hard': 1,
                          'soft': 2,
                          'others': 3}
        # build train set
        train_ids = []
        for anno in train_annotation_split:
            # pdb.set_trace()
            material = materials_dict[anno['material']]
            img_path = osp.join(self.dataset_dir, anno['image_path'].replace('\\', '/'))
            identity = int(anno['image_name'].split('_')[0])
            identity = identity - len(query_ids)
            # camera_id = int(anno['image_name'].split('_')[-1].split('.jpg')[0])
            # camera_id may be duplicate
            if anno['datatype'] == 'p':
                # camera_id = int(anno['image_name'].split('_')[-1].split('.jpg')[0])
                camera_id = 0
            else:
                # camera_id = int(anno['image_name'].split('_')[-1].split('.jpg')[0]) + 4
                camera_id = 1

            included_anno_train.append((img_path, identity, camera_id, material))
            train_ids.append(identity)

        # pdb.set_trace()
        return included_anno_train, included_anno_gallery, included_anno_query


    def build_testingset_by_spliting_traingset(self, test_annotation_split):
        included_anno_query = []  # test query set
        included_anno_gallery = []  # test gallery set

        # build test set
        gallery_ids = []
        for anno in test_annotation_split:
            # pdb.set_trace()
            img_path = osp.join(self.dataset_dir, anno['image_path'].replace('\\', '/'))

            if '_g_' in img_path:
                identity = int(anno['image_name'].split('_')[0])
                camera_id = int(anno['image_name'].split('_')[-1].split('.jpg')[0])
                included_anno_gallery.append((img_path, identity, camera_id))
                gallery_ids.append(identity)
            elif '_p_' in img_path:
                identity = int(anno['image_name'].split('_')[0])
                camera_id = int(anno['image_name'].split('_')[-1].split('.jpg')[0])
                included_anno_query.append((img_path, identity, camera_id))
            else:
                print('confusing image name with neither "g" nor "p".')

        query_ids = []
        filtered_anno_query = []
        for query in included_anno_query:
            _, q_id, _ = query
            if q_id not in gallery_ids:
                pass
            else:
                if q_id not in query_ids:
                    filtered_anno_query.append(query)
                    query_ids.append(q_id)
                else:
                    pass

        self.check_query_and_gallery(query_ids, gallery_ids)

        return filtered_anno_query, included_anno_gallery, query_ids

    def read_annotations_test_q(self, dir):
        with open(dir, 'r') as input_json:
            train_annotation = json.load(input_json)

        included_anno = []
        # bag_ids = []
        for anno in train_annotation['image']:
            # pdb.set_trace()
            img_path = osp.join(self.dataset_dir, anno['image_path'].replace('\\', '/'))
            # included_anno.append((img_path, anno['image_id'], int(anno['image_name'].split('.jpg')[0])))
            identity = (anno['image_name'].split('.jpg')[0])

            bag_id = int(anno['image_id'])
            included_anno.append((img_path, identity, 0))

        return included_anno

    def read_annotations_test_g(self, dir):
        with open(dir, 'r') as input_json:
            train_annotation = json.load(input_json)

        included_anno = []
        for anno in train_annotation['image']:
            img_path = osp.join(self.dataset_dir, anno['image_path'].replace('\\', '/'))
            identity = (anno['image_name'].split('_')[0])

            bag_id = int(anno['image_id'])
            included_anno.append((img_path, identity, 0))
        return included_anno

    def check_query_and_gallery(self, query, gallery):
        unincluded_queries = []
        for q in query:
            if q not in gallery:
                unincluded_queries.append(q)
        # pdb.set_trace()
        print('query {} are not included in gallery set'.format(unincluded_queries))