import json
from pycocotools.coco import COCO
from detectron2.evaluation.fast_eval_api import COCOeval_opt
import copy
import numpy as np
import os


json_root = os.path.join(os.environ['HOME'], 'work_dir/detectron2/projects/LidarSup/annotations')
nus_without_segm_path = os.path.join(json_root, 'nuscene_v1.0-mini_val_v02.json')
pseudo_segm_path = os.path.join(json_root, 'nuscenes_mini.segm.json')


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
    if iou == []:
        print(i, img_id, cat_id, 'no pseudo gt mask')
        valid_ann_flag = False
        valid_flag_list.append(valid_ann_flag)
        continue

    match_inds = np.argmax(iou[:,0], axis=0)
    ious[img_id, cat_id] = ious[img_id, cat_id][:, 1:]

    if pseudo_gt_ann[match_inds]['score'] < 0.5:
        valid_ann_flag = False
        valid_flag_list.append(valid_ann_flag)
        continue

    gt_ann['segmentation'] = pseudo_gt_ann[match_inds]['segmentation']
    gt_ann['area'] = pseudo_gt_ann[match_inds]['area']
    valid_flag_list.append(valid_ann_flag)

assert len(valid_flag_list) == len(gt_annos)
gt_annos = [gt_annos[i] for i in range(len(gt_annos)) if valid_flag_list[i]]

print('filter out {} ann without valid segmentation(score > 0.5), {} ann remaining.'.format(len(valid_flag_list)-len(gt_annos), len(gt_annos)))

coco_gt.dataset['annotations'] = gt_annos
save_path = os.path.join(json_root, 'nuscene_v1.0-mini_val_balance_with_maskv02.json')
with open(save_path, 'w') as f:
    json.dump(coco_gt.dataset, f)