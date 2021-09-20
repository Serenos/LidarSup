'''
Time:   2021/9/20 
Author: Lixiang
E-mail: 1075099620@qq.com / 3120211007@bit.edu.cn
Project: https://github.com/Serenos/LidarSup
'''

import json
from pycocotools.coco import COCO
from detectron2.evaluation.fast_eval_api import COCOeval_opt
import pycocotools.mask as mask_util
import copy
import numpy as np
import os


json_root = os.path.join(os.environ['HOME'], 'work_dir/detectron2/projects/LidarSup/annotations')
nus_without_segm_path = os.path.join(json_root, 'nuscene_v1.0-mini_val_v01.json')
pseudo_segm_path = os.path.join(json_root, 'nuscenes_mini.segm.json')

# 该版本 要保留nuscens原始数据的所有bbox，即使没有segm
bitmask = np.zeros((900,1600), order='F', dtype='uint8')
encode_mask = mask_util.encode(bitmask)
if isinstance(encode_mask['counts'], bytes):
    encode_mask['counts'] = encode_mask['counts'].decode()

#print(len(pseudo_gt_json), pseudo_gt_json[0])
coco_gt = COCO(nus_without_segm_path)
for ann in coco_gt.anns:
    coco_gt.anns[ann]['iscrowd'] = 0
coco_dt = coco_gt.loadRes(pseudo_segm_path)

iou_type = 'bbox'
coco_eval = COCOeval_opt(coco_gt, coco_dt, iou_type)
p = coco_eval.params
p.imgIds = list(np.unique(p.imgIds))
p.catIds = list(np.unique(p.catIds))

coco_eval.params = p
coco_eval._prepare()
catIds = p.catIds
computeIoU = coco_eval.computeIoU
coco_eval.ious = {
    (imgId, catId): computeIoU(imgId, catId) for imgId in p.imgIds for catId in catIds
}
ious = copy.deepcopy(coco_eval.ious)

gt_annos = coco_gt.dataset['annotations']
valid_flag_list = []

for i, gt_ann in enumerate(gt_annos):
    valid_ann_flag = True
    cat_id, img_id, ann_id = gt_ann['category_id'], gt_ann['image_id'], gt_ann['id']
    pseudo_gt_ann = coco_eval._dts[img_id, cat_id]
    iou = ious[img_id, cat_id]
    if len(iou) == 0:
        #print(i, img_id, cat_id, 'no pseudo gt mask')
        valid_ann_flag = False
        valid_flag_list.append(valid_ann_flag)
        gt_ann['segmentation'] = encode_mask
        gt_ann['area'] = 0
        continue

    match_inds = np.argmax(iou[:,0], axis=0)
    ious[img_id, cat_id] = ious[img_id, cat_id][:, 1:]

    if pseudo_gt_ann[match_inds]['score'] < 0.5:
        valid_ann_flag = False
        valid_flag_list.append(valid_ann_flag)
        gt_ann['segmentation'] = encode_mask
        gt_ann['area'] = 0
        continue
    if valid_ann_flag:
        gt_ann['segmentation'] = pseudo_gt_ann[match_inds]['segmentation']
        gt_ann['area'] = pseudo_gt_ann[match_inds]['area']
    valid_flag_list.append(valid_ann_flag)

# assert len(valid_flag_list) == len(gt_annos)
# gt_annos = [gt_annos[i] for i in range(len(gt_annos)) if valid_flag_list[i]]

print('{} ann with valid segmentation(score > 0.5), {} ann remaining.'.format(np.array(valid_flag_list).sum(), len(gt_annos)))

coco_gt.dataset['annotations'] = gt_annos
save_path = os.path.join(json_root, 'nuscene_v1.0-mini_val_balance_with_maskv01.json')
with open(save_path, 'w') as f:
    json.dump(coco_gt.dataset, f)