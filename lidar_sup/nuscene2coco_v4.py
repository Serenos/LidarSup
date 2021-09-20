import copy
import json
import numpy as np
import os
import sys
import random
import tqdm

from pyquaternion import Quaternion
from PIL import Image

try:
    from nuscenes.nuscenes import NuScenes
    from nuscenes.utils import splits
    from nuscenes.utils.data_classes import LidarPointCloud
    from nuscenes.utils.geometry_utils import view_points, points_in_box

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

#sample num for a instance
SAMPLE_NUM = 20

# using cam and lidar data
CAM_SENSOR = ['CAM_FRONT', 'CAM_FRONT_RIGHT', 'CAM_BACK_RIGHT', 'CAM_BACK',  'CAM_BACK_LEFT', 'CAM_FRONT_LEFT']
POINT_SENSOR = 'LIDAR_TOP'

def remove_overlap(depth_img):
    """
    # remove the overlap points in the projected image
    """
    k_size = 20#30
    hor_step = 10#10
    ver_step = 5
    height, width = depth_img.shape
    thresh_ratio = 0.2 #0.25

    for i in range(200, height, ver_step):
        for j in range(0, width, hor_step):
            if i >= height-k_size-1 or j >= width-k_size-1:
                continue
            value = depth_img[i:i+k_size, j:j+k_size]
            point_idx = np.where(value != 0)

            if len(point_idx[0]) <= 1:
                continue
            else:
                point_loc_list, depth_list = [], []
                point_i = point_idx[0] + i
                point_j = point_idx[1] + j
                for k in range(len(point_j)):
                    depth_list.append(depth_img[point_i[k], point_j[k]])
                    point_loc_list.append((point_i[k], point_j[k]))
                depth_list = np.array(depth_list)
                point_loc_list = np.array(point_loc_list)
                depth_min = depth_list.min()
                depth_max = depth_list.max()
                if (depth_max - depth_min) / depth_min < thresh_ratio:
                    continue
                min_idx = np.where(value == depth_min)
                if min_idx[0][0] == k_size-1:
                    continue

                depth_minus = (depth_list - depth_min) / depth_min
                idx_near = np.where(depth_minus < thresh_ratio)[0]
                idx_far = np.where(depth_minus >= thresh_ratio)[0]

                if len(idx_near) > 1:
                    pix_list = point_loc_list[idx_near]
                    hor_min = pix_list[:, 1].min() - k_size/2
                    hor_max = pix_list[:, 1].max() + k_size/2
                    ver_min = pix_list[:, 0].min() - 1
                else:
                    pix_list = point_loc_list[idx_near]
                    hor_min = pix_list[:, 1][0] - k_size/2
                    hor_max = hor_min + k_size
                    ver_min = pix_list[:, 0][0] - 1

                for p in range(len(idx_far)):
                    if point_loc_list[idx_far[p]][1] >= hor_min and\
                        point_loc_list[idx_far[p]][1] <= hor_max:
                        #point_loc_list[idx_far[p]][0] >= ver_min:
                        depth_img[point_loc_list[idx_far[p]][0], point_loc_list[idx_far[p]][1]] = 0.

    return depth_img

def map_point_to_img(nusc, pc, pointsensor, cam):
    img = Image.open(os.path.join(nusc.dataroot, cam['filename']))

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
    mask = np.logical_and(mask, points[0, :] > 1)
    mask = np.logical_and(mask, points[0, :] < img.size[0] - 1)
    mask = np.logical_and(mask, points[1, :] > 1)
    mask = np.logical_and(mask, points[1, :] < img.size[1] - 1)
    points = points[:, mask].astype(np.int16)
    coloring = coloring[mask]

    return points, coloring, img, mask

def filter_with_2dbox(points, h, w, depth, bbox=None):
    mask = np.ones(points.shape[1], dtype=bool)
    if bbox:
        mask = np.logical_and(mask, depth > 1)
        mask = np.logical_and(mask, points[0, :] > max(bbox[0]+1, 1))
        mask = np.logical_and(mask, points[0, :] < min(bbox[0] + bbox[2]-1, w - 1))
        mask = np.logical_and(mask, points[1, :] > max(bbox[1]+1, 1))
        mask = np.logical_and(mask, points[1, :] < min(bbox[1] + bbox[3]-1, h - 1))
    else:
        mask = np.logical_and(mask, depth > 1)
        mask = np.logical_and(mask, points[0, :] > 1)
        mask = np.logical_and(mask, points[0, :] < w - 1)
        mask = np.logical_and(mask, points[1, :] > 1)
        mask = np.logical_and(mask, points[1, :] < h - 1)
    points = points[:2, mask]
    depth = depth[mask]
    return points ,depth

def random_sample(sample_num, pc_inbox, pc_outbox):
    pc_inbox_label = np.concatenate((pc_inbox, np.ones((1, pc_inbox.shape[1]))), axis=0)
    pc_outbox_label = np.concatenate((pc_outbox, np.zeros((1, pc_outbox.shape[1]))), axis=0)
    pc_with_label = np.concatenate((pc_inbox_label, pc_outbox_label), axis=1)
        
    a = np.arange(pc_with_label.shape[1])
    mask = random.sample(list(a), sample_num)
    point_coords = pc_with_label[:2, mask].T.tolist()
    point_label = pc_with_label[2, mask].T.tolist()
    return point_coords, point_label

def random_sample_balance(sample_num, pc_inbox, pc_outbox):
    pc_inbox_label = np.concatenate((pc_inbox, np.ones((1, pc_inbox.shape[1]))), axis=0)
    pc_outbox_label = np.concatenate((pc_outbox, np.zeros((1, pc_outbox.shape[1]))), axis=0)
    #print(pc_inbox_label.shape[1], pc_outbox_label.shape[1])
    a = np.arange(pc_inbox_label.shape[1])
    b = np.arange(pc_outbox_label.shape[1])
    mask_point_in_box = random.sample(list(a), 10)
    mask_point_out_box = random.sample(list(b), 10)
    pc_inbox_label = pc_inbox_label[:, mask_point_in_box]
    pc_outbox_label = pc_outbox_label[:, mask_point_out_box]
    pc_with_label = np.concatenate((pc_inbox_label, pc_outbox_label), axis=1).T
    np.random.shuffle(pc_with_label)
    point_coords = pc_with_label[:, :2].tolist()
    point_label = pc_with_label[:, 2].tolist()
    return point_coords, point_label

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
    scene = train_scenes if split=='train' else val_scenes
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
    for sample_index, sample_record in enumerate(tqdm.tqdm(nusc.sample)):
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
                'id': 6 * sample_index + index
            }
            new_coco_dataset['images'].append(coco_image)

        point_data_token = sample_record['data'][POINT_SENSOR]
        point_data = nusc.get('sample_data', point_data_token)
        lidar_path = point_data['filename']
        full_lidar_path = os.path.join(nusc.dataroot, lidar_path)
        pc = LidarPointCloud.from_file(full_lidar_path)
        
        # remove the overlap pc for 6 pictures of a sample
        refine_pc2d_dict = {}
        refine_pc3d_dict = {}
        refine_color_cit = {}
        for cam in CAM_SENSOR:
            camera_token = sample_record['data'][cam]
            cam_data = nusc.get('sample_data', camera_token)
            pc_temp = copy.deepcopy(pc)
            point2d, coloring, img, mask = map_point_to_img(nusc, pc_temp, point_data, cam_data)
            point3d = pc.points[:,mask]
            depth_map = np.zeros((img.size[1],img.size[0]))
            loc2index = np.zeros((img.size[1],img.size[0]))

            for i in range(point2d.shape[1]):
                depth_map[point2d[1,i], point2d[0,i]] = coloring[i]
                loc2index[point2d[1,i], point2d[0,i]] = int(i)
            refine_depth_map = copy.deepcopy(depth_map)
            refine_depth_map = remove_overlap(depth_img=refine_depth_map)
            total_point_num = point2d.shape[1]

            mask = np.ones(total_point_num)
            for i in range(img.size[1]):
                for j in range(img.size[0]):
                    if depth_map[i,j]>0 and refine_depth_map[i,j]==0:
                            point_index = loc2index[i,j]
                            #print(point_index)
                            mask[int(point_index)] = 0
            mask = mask.astype(np.bool8) 
            point2d, point3d, coloring = point2d[:, mask], point3d[:, mask], coloring[mask]
            #print('remove {} overlaping points'.format(total_point_num - point2d.shape[1]))
            refine_pc2d_dict[cam] = point2d
            refine_pc3d_dict[cam] = point3d
            refine_color_cit[cam] = coloring
            
        fliter_ann_num = 0
        sample_annotation_tokens = sample_record['anns']
        for j, sample_annotation_token in enumerate(sample_annotation_tokens):

            # Figure out which camera the object is fully visible in (this may return nothing).
            boxes, cam = [], []
            for cam in CAM_SENSOR:
                _, boxes, _ = nusc.get_sample_data(sample_record['data'][cam], box_vis_level=1, selected_anntokens=[sample_annotation_token])
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
            _, boxes, camera_intrinsic = nusc.get_sample_data(cam_data_token, box_vis_level=1, selected_anntokens=[box.token])
            box = boxes[0]

            #3D box to 2D box
            box_coord = view_points(box.corners(), camera_intrinsic, normalize=True)[:2, :]
            x = np.array([max(box_coord[0, :]), min(box_coord[0, :])])
            y = np.array([max(box_coord[1, :]), min(box_coord[1, :])])
            xtl, ytl, w, h = x[1] , y[1], (x[0] - x[1]), (y[0] - y[1]) 
            bbox = [xtl, ytl, w, h]

            category_id = CLASS2ID[NameMapping[box.name]]
            _, box_lidar_frame, _ = nusc.get_sample_data(point_data_token, selected_anntokens=[box.token])
            box_lidar_frame = box_lidar_frame[0]

            pc3d = refine_pc3d_dict[cam]
            pc2d = refine_pc2d_dict[cam]
            depth = refine_color_cit[cam]

            logits = points_in_box(box_lidar_frame, pc3d[:3,:])

            pc2d_inbox = copy.deepcopy(pc2d)
            pc2d_outbox = copy.deepcopy(pc2d)
            pc2d_inbox= pc2d_inbox[:, logits]
            pc2d_outbox = pc2d_outbox[:, ~logits]

            depth_inbox = copy.deepcopy(depth)
            depth_outbox = copy.deepcopy(depth)
            depth_inbox= depth_inbox[logits]
            depth_outbox = depth_outbox[~logits]


            pc2d_inbox, depth_inbox = filter_with_2dbox(pc2d_inbox, img.size[1], img.size[0], depth_inbox)
            pc2d_outbox, depth_outbox = filter_with_2dbox(pc2d_outbox, img.size[1], img.size[0], depth_outbox, bbox)
            # if pc2d_inbox.shape[1] + pc2d_outbox.shape[1] < SAMPLE_NUM:
            #     continue
            if pc2d_inbox.shape[1]  < SAMPLE_NUM/2 or pc2d_outbox.shape[1] < SAMPLE_NUM/2:
                fliter_ann_num += 1
                continue
            # random sample SAMPLE_NUM points for pos/neg 
            point_coords, point_label = random_sample_balance(20, pc2d_inbox, pc2d_outbox)

            coco_ann = {
                'area': w * h,
                'image_id': 6*sample_index + CAM_SENSOR.index(cam),
                'bbox': bbox,
                'category_id': category_id,
                'id': ann_index,
                'point_coords': point_coords,
                'point_labels': point_label 
            }
            ann_index += 1
            new_coco_dataset['annotations'].append(coco_ann)
        #print('fliter {}/{} annotations by lidar points in sample {}'.format(fliter_ann_num, j, sample_index))

    return new_coco_dataset



if __name__ == '__main__':
    version = sys.argv[1]
    split = sys.argv[2]
    available_vers = ['v1.0-trainval', 'v1.0-test', 'v1.0-mini']
    if version not in available_vers:
        raise ValueError('unknown version')
    if split not in ['train', 'val']:
        raise ValueError('unknown split')

    print('converting nuscens_{}_{} to coco format......'.format(version ,split))
    dataroot = os.path.join(os.environ['HOME'], 'datasets/nuscenes')
    new_coco_dataset = prepare_coco_data(version=version, split=split, dataroot=dataroot)

    input_filename = 'nuscenes'
    output_filename = 'nuscene_{}_{}_balance_without_mask{}.json'.format(version ,split, 'v4')
    project_path = os.path.join(os.environ['HOME'], 'lixiang/detectron2/projects/Lidarsup')
    json_root = os.path.join(project_path, 'json')
    out_path = os.path.join(json_root, output_filename)

    with open(out_path, 'w') as f:
        json.dump(new_coco_dataset, f)
    print("{} is modified and stored in {}.".format(input_filename, output_filename))
