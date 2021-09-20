import logging
import os
import json

from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.data.datasets.builtin import _get_builtin_metadata
from detectron2.data.datasets.coco import load_coco_json

# NuScenes dataset in coco format
def register_nuscenes_instances_with_points_and_box(name, json_file, image_root):
    """
    Register a dataset in COCO's json annotation format for
    instance segmentation with point annotation.

    The point annotation json does not have "segmentation" field, instead,
    it has "point_coords" and "point_labels" fields.

    Args:
        name (str): the name that identifies a dataset, e.g. "coco_2014_train".
        metadata (dict): extra metadata associated with this dataset.  You can
            leave it as an empty dict.
        json_file (str): path to the json instance annotation file.
        image_root (str or path-like): directory which contains all the images.
    """
    assert isinstance(name, str), name
    assert isinstance(json_file, (str, os.PathLike)), json_file
    assert isinstance(image_root, (str, os.PathLike)), image_root
    # 1. register a function which returns dicts
    DatasetCatalog.register(
        name, lambda: load_coco_json(json_file, image_root, name, ["bbox", "sample_token", "cam", "ann_token", "point_coords", "point_labels"])
    )

    # 2. Optionally, add metadata about this dataset,
    # since they might be useful in evaluation, visualization or logging
#     MetadataCatalog.get(name).set(
#         json_file=json_file, image_root=image_root, evaluator_type="coco", **metadata
#     )

_root = os.environ['HOME']
json_root = os.path.join(_root, 'work_dir/detectron2/projects/LidarSup/annotations')
img_root =os.path.join(_root, 'datasets/nuscenes')


# n_10_json_filev3 = 'sample_10points_without_maskv3.json'
# n_10_json_filev3_path = os.path.join(_root, n_10_json_filev3)
# register_nuscenes_instances_with_points_and_box('nuscenes_10v3', n_10_json_filev3_path, img_root)

# n_20_json_filev3 = 'sample_20points_without_maskv3.json'
# n_20_json_filev3_path = os.path.join(_root, n_20_json_filev3)
# register_nuscenes_instances_with_points_and_box('nuscenes_20v3', n_20_json_filev3_path, img_root)

# n_20_json_filev4 = 'sample_20points_without_maskv4.json'
# n_20_json_filev4_path = os.path.join(_root, n_20_json_filev4)
# register_nuscenes_instances_with_points_and_box('nuscenes_20v4', n_20_json_filev4_path, img_root)

# nuscene_mini_val_v4 = 'nuscene_v1.0-mini_val_without_maskv4.json'
# nuscene_mini_val_v4_path = os.path.join(_root, nuscene_mini_val_v4)
# register_nuscenes_instances_with_points_and_box('nuscenes_mini_val_v4', nuscene_mini_val_v4_path, img_root)


json_train02 = 'nuscene_v1.0-mini_val_v02.json'
json_val02 = 'nuscene_v1.0-mini_val_balance_with_maskv02.json'
json_train02_path = os.path.join(json_root, json_train02)
json_val02_path = os.path.join(json_root, json_val02)
register_nuscenes_instances_with_points_and_box('nuscene_v1.0-mini_val_v02', json_train02_path, img_root)
register_nuscenes_instances_with_points_and_box('nuscene_v1.0-mini_val_balance_with_maskv02', json_val02_path, img_root)

json_train01 = 'nuscene_v1.0-mini_val_v01.json'
json_val01 = 'nuscene_v1.0-mini_val_balance_with_maskv01.json'
json_train01_path = os.path.join(json_root, json_train01)
json_val01_path = os.path.join(json_root, json_val01)
register_nuscenes_instances_with_points_and_box('nuscene_v1.0-mini_val_v01', json_train01_path, img_root)
register_nuscenes_instances_with_points_and_box('nuscene_v1.0-mini_val_balance_with_maskv01', json_val01_path, img_root)

# json_train1 = 'nuscene_v1.0-mini_train_balance_without_maskv4.json'
# json_val1 = 'nuscene_v1.0-mini_val_balance_with_maskv4.json'
# json_train1_path = os.path.join(json_root, json_train1)
# json_val1_path = os.path.join(json_root, json_val1)
# register_nuscenes_instances_with_points_and_box('nuscene_v1.0-mini_train_balance_without_maskv4', json_train1_path, img_root)
# register_nuscenes_instances_with_points_and_box('nuscene_v1.0-mini_val_balance_with_maskv4', json_val1_path, img_root)
#
# json_train2 = 'nuscene_v1.0-mini_train_balance_without_maskv3.json'
# json_val2 = 'nuscene_v1.0-mini_val_balance_with_maskv3.json'
# json_train2_path = os.path.join(json_root, json_train2)
# json_val2_path = os.path.join(json_root, json_val2)
# register_nuscenes_instances_with_points_and_box('nuscene_v1.0-mini_train_balance_without_maskv3', json_train2_path, img_root)
# register_nuscenes_instances_with_points_and_box('nuscene_v1.0-mini_val_balance_with_maskv3', json_val2_path, img_root)