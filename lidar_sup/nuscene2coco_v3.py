import copy
import json
import numpy as np
import os
import sys
import random
import tqdm
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib import rcParams
from matplotlib.axes import Axes

from pyquaternion import Quaternion
from PIL import Image
from torch.utils import data

try:
    from nuscenes.nuscenes import NuScenes
    from nuscenes.utils import splits
    from nuscenes.utils.splits import create_splits_scenes
    from nuscenes.utils.data_classes import LidarPointCloud, RadarPointCloud, Box
    from nuscenes.utils.geometry_utils import view_points, box_in_image, points_in_box, BoxVisibility, transform_matrix

except:
    print("nuScenes devkit not Found!")

NameMapping = {
    'movable_object.barrier': 'barrier',
    'vehicle.bicycle': 'bicycle',
    'vehicle.bus.bendy': 'bus',
    'vehicle.bus.rigid': 'bus',
    'vehicle.car': 'car',
    'vehicle.construction': 'construction_vehicle',
    'vehicle.motorcycle': 'motorcycle',
    'human.pedestrian.adult': 'pedestrian',
    'human.pedestrian.child': 'pedestrian',
    'human.pedestrian.construction_worker': 'pedestrian',
    'human.pedestrian.police_officer': 'pedestrian',
    'movable_object.trafficcone': 'traffic_cone',
    'vehicle.trailer': 'trailer',
    'vehicle.truck': 'truck'
}

CLASS2ID = {
    'car': 0,
    'truck': 1,
    'trailer': 2,
    'bus': 3,
    'construction_vehicle': 4,
    'bicycle': 5,
    'motorcycle': 6,
    'pedestrian': 7,
    'traffic_cone': 8,
    'barrier': 9
}

CLASSES = ('car', 'truck', 'trailer', 'bus', 'construction_vehicle',
           'bicycle', 'motorcycle', 'pedestrian', 'traffic_cone',
           'barrier')

CAM_SENSOR = ['CAM_FRONT', 'CAM_FRONT_RIGHT', 'CAM_BACK_RIGHT', 'CAM_BACK', 'CAM_BACK_LEFT', 'CAM_FRONT_LEFT']
POINT_SENSOR = 'LIDAR_TOP'


def view_point_in_box(nusc, pc, pointsensor, cam, w, h):
    # img = Image.open(os.path.join(nusc.dataroot, cam['filename']))

    # projection
    cs_record = nusc.get('calibrated_sensor', pointsensor['calibrated_sensor_token'])
    pc.rotate(Quaternion(cs_record['rotation']).rotation_matrix)
    pc.translate(np.array(cs_record['translation']))

    pose_record = nusc.get('ego_pose', pointsensor['ego_pose_token'])
    pc.rotate(Quaternion(pose_record['rotation']).rotation_matrix)
    pc.translate(np.array(pose_record['translation']))

    pose_record = nusc.get('ego_pose', cam['ego_pose_token'])
    pc.translate(-np.array(pose_record['translation']))
    pc.rotate(Quaternion(pose_record['rotation']).rotation_matrix.T)

    cs_record = nusc.get('calibrated_sensor', cam['calibrated_sensor_token'])
    pc.translate(-np.array(cs_record['translation']))
    pc.rotate(Quaternion(cs_record['rotation']).rotation_matrix.T)

    depths = pc.points[2, :]
    coloring = depths

    points = view_points(pc.points[:3, :], np.array(cs_record['camera_intrinsic']), normalize=True)
    cs_record['camera_intrinsic']

    mask = np.ones(depths.shape[0], dtype=bool)
    mask = np.logical_and(mask, depths > 1)
    mask = np.logical_and(mask, points[0, :] > 1)
    mask = np.logical_and(mask, points[0, :] < w - 1)
    mask = np.logical_and(mask, points[1, :] > 1)
    mask = np.logical_and(mask, points[1, :] < h - 1)
    points = points[:, mask]
    coloring = coloring[mask]

    return points, depths

def view_point_out_box(nusc, bbox, pc, pointsensor, cam, w, h):
    # img = Image.open(os.path.join(nusc.dataroot, cam['filename']))

    # projection
    cs_record = nusc.get('calibrated_sensor', pointsensor['calibrated_sensor_token'])
    pc.rotate(Quaternion(cs_record['rotation']).rotation_matrix)
    pc.translate(np.array(cs_record['translation']))

    pose_record = nusc.get('ego_pose', pointsensor['ego_pose_token'])
    pc.rotate(Quaternion(pose_record['rotation']).rotation_matrix)
    pc.translate(np.array(pose_record['translation']))

    pose_record = nusc.get('ego_pose', cam['ego_pose_token'])
    pc.translate(-np.array(pose_record['translation']))
    pc.rotate(Quaternion(pose_record['rotation']).rotation_matrix.T)

    cs_record = nusc.get('calibrated_sensor', cam['calibrated_sensor_token'])
    pc.translate(-np.array(cs_record['translation']))
    pc.rotate(Quaternion(cs_record['rotation']).rotation_matrix.T)

    depths = pc.points[2, :]
    coloring = depths

    camera_intrinsic = np.array(cs_record['camera_intrinsic'])
    points = view_points(pc.points[:3, :], camera_intrinsic, normalize=True)

    mask = np.ones(depths.shape[0], dtype=bool)
    mask = np.logical_and(mask, depths > 1)
    mask = np.logical_and(mask, points[0, :] > max(bbox[0] + 1, 1))
    mask = np.logical_and(mask, points[0, :] < min(bbox[0] + bbox[2] - 1, w - 1))
    mask = np.logical_and(mask, points[1, :] > max(bbox[1] + 1, 1))
    mask = np.logical_and(mask, points[1, :] < min(bbox[1] + bbox[3] - 1, h - 1))
    points = points[:, mask]
    coloring = coloring[mask]

    return points, depths


def refine_lidar_points(point_in, point_out):
    m, n = point_in.shape[1], point_out.shape[1]
    # print(m,n)
    X, Y = point_in[:2, :], point_out[:2, :]
    G = np.dot(X.T, Y)
    H = np.tile(np.diag(np.dot(X.T, X)), (n, 1)).T
    K = np.tile(np.diag(np.dot(Y.T, Y)), (m, 1))
    D = H + K - 2 * G
    # print(D.shape)
    D = np.min(D, axis=0)
    mask = D > 169
    Y = Y[:, mask]
    return Y

def split_scene(nusc, version):
    if version == 'v1.0-trainval':
        train_scenes = splits.train
        val_scenes = splits.val
    elif version == 'v1.0-test':
        train_scenes = splits.test
        val_scenes = []
    elif version == 'v1.0-mini':
        train_scenes = splits.mini_train
        val_scenes = splits.mini_val
    else:
        raise ValueError('unknown')

    scene_names = [s['name'] for s in nusc.scene]
    train_scenes = list(filter(lambda x: x in scene_names, train_scenes))
    val_scenes = list(filter(lambda x: x in scene_names, val_scenes))
    train_scenes = set([
            nusc.scene[scene_names.index(s)]['token']
            for s in train_scenes
        ])
    val_scenes = set([
            nusc.scene[scene_names.index(s)]['token']
            for s in val_scenes
        ])
    print('train_scenes: ', len(train_scenes))
    print('val_scenes: ', len(val_scenes))
    return train_scenes, val_scenes


def prepare_coco_data(version='v1.0-mini', split='val',dataroot='/home/PJLAB/lixiang/datasets/nuscenes'):
    nusc = NuScenes(version=version, dataroot=dataroot)
    train_scenes, val_scenes = split_scene(nusc, version)
    scene = train_scenes if split == 'train' else val_scenes
    # create coco_data for nuScenes
    new_coco_dataset = {}
    # -COCO_CATEGORY
    new_coco_dataset['categories'] = []
    for i, name in enumerate(CLASSES):
        new_coco_dataset['categories'].append(
            {
                'id': i,
                'name': name,
                'supercategory': 'mark'
            }
        )
    new_coco_dataset['images'] = []
    new_coco_dataset['annotations'] = []
    ann_index = 0
    for i, sample_record in enumerate(tqdm.tqdm(nusc.sample)):
        if sample_record['scene_token'] not in scene:
            continue
        # -COCO_IMG: a sample includes 6 img from different CAM_SENSOR
        for index, cam in enumerate(CAM_SENSOR):
            cam_data_token = sample_record['data'][cam]
            cam_data = nusc.get('sample_data', cam_data_token)
            img_path = cam_data['filename']
            full_img_path = os.path.join(nusc.dataroot, img_path)
            height, width = cam_data['height'], cam_data['width']

            coco_image = {
                'file_name': img_path,
                'height': height,
                'width': width,
                'id': 6 * i + index
            }
            new_coco_dataset['images'].append(coco_image)

        point_data_token = sample_record['data'][POINT_SENSOR]
        point_data = nusc.get('sample_data', point_data_token)
        lidar_path = point_data['filename']
        full_lidar_path = os.path.join(nusc.dataroot, lidar_path)
        pc = LidarPointCloud.from_file(full_lidar_path)

        sample_annotation_tokens = sample_record['anns']

        for j, sample_annotation_token in enumerate(sample_annotation_tokens):
            ann_record = nusc.get('sample_annotation', sample_annotation_token)
            if ann_record['num_lidar_pts'] < 10:
                continue
            assert 'LIDAR_TOP' in sample_record['data'].keys(), 'Error: No LIDAR_TOP in data'

            # Figure out which camera the object is fully visible in (this may return nothing).
            boxes, cam = [], []
            cams = [key for key in sample_record['data'].keys() if 'CAM' in key]
            for cam in cams:
                _, boxes, _ = nusc.get_sample_data(sample_record['data'][cam], box_vis_level=1,
                                                   selected_anntokens=[sample_annotation_token])
                if len(boxes) > 0:
                    break  # We found an image that matches. Let's abort.
            # assert len(boxes) > 0, 'Error: Could not find image where annotation is visible. ' \
            #                     'Try using e.g. BoxVisibility.ANY.'
            # assert len(boxes) < 2, 'Error: Found multiple annotations. Something is wrong!'
            if len(boxes) >= 2 or len(boxes) <= 0:
                continue

            cam_data_token = sample_record['data'][cam]

            # filter annotations box not in the 10 categories
            box = boxes[0]
            if box.name not in NameMapping:
                continue
            _, boxes, camera_intrinsic = nusc.get_sample_data(cam_data_token, box_vis_level=1,
                                                              selected_anntokens=[box.token])
            box = boxes[0]

            # 3D box to 2D box
            box_coord = view_points(box.corners(), camera_intrinsic, normalize=True)[:2, :]
            x = np.array([max(box_coord[0, :]), min(box_coord[0, :])])
            y = np.array([max(box_coord[1, :]), min(box_coord[1, :])])
            #corners2d = np.vstack((x, y))
            # xyxy mode to xyhw mode
            xtf = x[1]
            ytf = y[1]
            # xc = (x[0] + x[1]) / 2.0
            # yc = (y[0] + y[1]) / 2.0
            w = (x[0] - x[1])
            h = (y[0] - y[1])
            bbox = [xtf, ytf, w, h]

            category_id = CLASS2ID[NameMapping[box.name]]

            _, box_lidar_frame, _ = nusc.get_sample_data(point_data_token, selected_anntokens=[box.token])
            box_lidar_frame = box_lidar_frame[0]
            logits = points_in_box(box_lidar_frame, pc.points[:3, :])

            pc_inbox = copy.deepcopy(pc)
            pc_outbox = copy.deepcopy(pc)
            pc_inbox.points = pc_inbox.points[:, logits]
            pc_outbox.points = pc_outbox.points[:, ~logits]

            cam_data = nusc.get('sample_data', cam_data_token)
            point_in_box, _ = view_point_in_box(nusc, pc_inbox, point_data, cam_data, width, height)
            point_out_box, _ = view_point_out_box(nusc, bbox, pc_outbox, point_data, cam_data, width, height)
            if point_in_box.shape[1] < 10 or point_out_box.shape[1] < 10:
                continue
            point_out_box = refine_lidar_points(point_in_box, point_out_box)
            if point_in_box.shape[1] < 10 or point_out_box.shape[1] < 10:
                continue

            # random sample 5/5 points for pos/neg
            a = np.arange(point_in_box.shape[1])
            b = np.arange(point_out_box.shape[1])
            mask_point_in_box = random.sample(list(a), 10)
            mask_point_out_box = random.sample(list(b), 10)
            sample_lidar_point_in_box = point_in_box[:, mask_point_in_box]
            sample_lidar_point_out_box = point_out_box[:, mask_point_out_box]
            S1 = sample_lidar_point_in_box[:2, :]
            S2 = sample_lidar_point_out_box[:2, :]
            S_finale = np.concatenate((S1, S2), axis=1).T
            point_label = np.array([[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]]).T

            # shuffle
            coord_label = np.concatenate((S_finale, point_label), axis=1)
            np.random.shuffle(coord_label)
            coord_shuffle = coord_label[:, :2].tolist()
            label_shuffle = coord_label[:, 2].tolist()
            # sample_depths = depths[mask_point_in_box] # To Do how to use intensity information
            coco_ann = {
                'area': w * h,
                'image_id': 6 * i + CAM_SENSOR.index(cam),
                'bbox': bbox,
                'category_id': category_id,
                'id': ann_index,
                'point_coords': coord_shuffle,
                'point_labels': label_shuffle
            }
            ann_index += 1
            new_coco_dataset['annotations'].append(coco_ann)
            # print('----ann_index: {}, category: {}, points_num: {}'.format(ann_index, box.name, pc_temp.points.shape[1]))

    return new_coco_dataset


if __name__ == '__main__':
    version = sys.argv[1]
    split = sys.argv[2]
    available_vers = ['v1.0-trainval', 'v1.0-test', 'v1.0-mini']
    if version not in available_vers:
        raise ValueError('unknown version')
    if split not in ['train', 'val']:
        raise ValueError('unknown split')

    print('converting nuscens_{}_{} to coco format......'.format(version, split))
    dataroot = os.path.join(os.environ['HOME'], 'datasets/nuscenes')
    new_coco_dataset = prepare_coco_data(version=version, split=split, dataroot=dataroot)

    input_filename = 'nuscenes'
    output_filename = 'nuscene_{}_{}_balance_without_mask{}.json'.format(version, split, 'v3')
    project_path = os.path.join(os.environ['HOME'], 'lixiang/detectron2/projects/LidarSup')

    json_root = os.path.join(project_path, 'json')
    out_path = os.path.join(json_root, output_filename)

    with open(out_path, 'w') as f:
        json.dump(new_coco_dataset, f)
    print("{} is modified and stored in {}.".format(input_filename, output_filename))