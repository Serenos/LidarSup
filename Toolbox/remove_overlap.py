##
# Copyright (C) 2021-2022 by Shanghai AI Lab. All rights reserved.
# Ma Tao <matao@pjlab.org.cn>
##

import numpy as np


def remove_overlap(depth_img):
    """
    # remove the overlap points in the projected image
    input: depth_img -> a depth_map of size (h, w), pixel filled with point depth, otherwise 0.
    output: depth_img -> a depth map removing overlap pointclouds. 
    """
    k_size = 15
    hor_step = 10
    ver_step = 5
    height, width = depth_img.shape
    thresh_ratio = 0.25

    for i in range(400, height, ver_step):
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
                    hor_min = pix_list[:, 1].min() -1
                    hor_max = pix_list[:, 1].max() + 1
                    ver_min = pix_list[:, 0].min() - 1
                else:
                    pix_list = point_loc_list[idx_near]
                    hor_min = pix_list[:, 1][0] - 1
                    hor_max = hor_min + 2
                    ver_min = pix_list[:, 0][0] - 1

                for p in range(len(idx_far)):
                    if point_loc_list[idx_far[p]][1] >= hor_min and\
                        point_loc_list[idx_far[p]][1] <= hor_max and\
                        point_loc_list[idx_far[p]][0] >= ver_min:
                        depth_img[point_loc_list[idx_far[p]][0], point_loc_list[idx_far[p]][1]] = 0.

    return depth_img

