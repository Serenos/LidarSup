'''
Time:   2021/9/20 下午5:28
Author: Lixiang
E-mail: 1075099620@qq.com / 3120211007@bit.edu.cn
Project: https://github.com/Serenos/LidarSup
'''

import argparse
import json
import numpy as np
import os
import cv2
import tqdm

from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.utils.visualizer import Visualizer
from detectron2.projects.lidar_sup import register_lidar_annotations

from nuscenes.nuscenes import NuScenes
from nuscenes.utils.geometry_utils import view_points

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="a map from nuscenes_mask.json to origin nuscenese datasets.")
    parser.add_argument("--version", help="version of the nuscenes: 'v1.0-trainval', 'v1.0-test', 'v1.0-mini']", default="v1.0-mini")
    parser.add_argument("--dataset", help="name of the dataset", default="nuscene_v1.0-mini_val_balance_with_maskv01")
    args = parser.parse_args()

    dicts = list(DatasetCatalog.get(args.dataset))
    metadata = MetadataCatalog.get(args.dataset)

    dataroot = os.path.join(os.environ['HOME'], 'datasets/nuscenes')
    nusc = NuScenes(version=args.version, dataroot=dataroot)

    for i, dict in enumerate(dicts):
        #img = cv2.imread(dict["file_name"], cv2.IMREAD_COLOR)[:, :, ::-1]

        if len(dict['annotations']) == 0:
            continue
        cam = dict['annotations'][0]['cam']
        sample_token = dict['annotations'][0]['sample_token']

        sample_record = nusc.get('sample', sample_token)
        cam_data_token = sample_record['data'][cam]
        cam_data = nusc.get('sample_data', cam_data_token)
        img_path = os.path.join(dataroot, cam_data['filename'])
        #nus_img = cv2.imread(img_path, cv2.IMREAD_COLOR)[:, :, ::-1]
        if os.path.basename(dict["file_name"]) == os.path.basename(img_path):
            print('find the same img in nuscenes datasets through json file.')


        for j, ann in enumerate(dict['annotations']):

            ann_token = ann['ann_token']
            _, nus_box, camera_intrinsic = nusc.get_sample_data(cam_data_token, selected_anntokens=[ann_token])
            box_coord = view_points(nus_box[0].corners(), camera_intrinsic, normalize=True)[:2, :]
            x = np.array([max(box_coord[0, :]), min(box_coord[0, :])])
            y = np.array([max(box_coord[1, :]), min(box_coord[1, :])])
            xtf, ytf = x[1], y[1]
            w, h = (x[0] - x[1]), (y[0] - y[1])
            nus_box = [xtf, ytf, w, h]
