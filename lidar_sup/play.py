import json
from pycocotools.coco import COCO
import os


json_root = os.path.join(os.environ['HOME'], 'work_dir/detectron2/projects/LidarSup/annotations')
nus_without_segm_path = os.path.join(json_root, 'nuscene_v1.0-mini_val_balance_with_maskv01.json')
coco_gt = COCO(nus_without_segm_path)
for ann in coco_gt.anns:
    coco_gt.anns[ann]['iscrowd'] = 0

count = 0
for i, gt_ann in enumerate(coco_gt.dataset['annotations']):
    segm = gt_ann['segmentation']


print('missing {} mask.'.format(count))