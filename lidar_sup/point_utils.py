# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import torch

from detectron2.layers import cat

from skimage import color
import torch.nn.functional as F

def get_point_coords_from_point_annotation(instances):
    """
    Load point coords and their corresponding labels from point annotation.

    Args:
        instances (list[Instances]): A list of N Instances, where N is the number of images
            in the batch. These instances are in 1:1
            correspondence with the pred_mask_logits. The ground-truth labels (class, box, mask,
            ...) associated with each instance are stored in fields.
    Returns:
        point_coords (Tensor): A tensor of shape (N, P, 2) that contains the coordinates of P
            sampled points.
        point_labels (Tensor): A tensor of shape (N, P) that contains the labels of P
            sampled points. `point_labels` takes 3 possible values:
            - 0: the point belongs to background
            - 1: the point belongs to the object
            - -1: the point is ignored during training
    """
    point_coords_list = []
    point_labels_list = []
    for instances_per_image in instances:
        if len(instances_per_image) == 0:
            continue
        point_coords = instances_per_image.gt_point_coords.to(torch.float32)
        point_labels = instances_per_image.gt_point_labels.to(torch.float32).clone()
        proposal_boxes_per_image = instances_per_image.proposal_boxes.tensor

        # Convert point coordinate system, ground truth points are in image coord.
        point_coords_wrt_box = get_point_coords_wrt_box(proposal_boxes_per_image, point_coords)

        # Ignore points that are outside predicted boxes.
        point_ignores = (
            (point_coords_wrt_box[:, :, 0] < 0)
            | (point_coords_wrt_box[:, :, 0] > 1)
            | (point_coords_wrt_box[:, :, 1] < 0)
            | (point_coords_wrt_box[:, :, 1] > 1)
        )
        point_labels[point_ignores] = -1

        point_coords_list.append(point_coords_wrt_box)
        point_labels_list.append(point_labels)

    return (
        cat(point_coords_list, dim=0),
        cat(point_labels_list, dim=0),
    )


def get_point_coords_wrt_box(boxes_coords, point_coords):
    """
    Convert image-level absolute coordinates to box-normalized [0, 1] x [0, 1] point cooordinates.
    Args:
        boxes_coords (Tensor): A tensor of shape (R, 4) that contains bounding boxes.
            coordinates.
        point_coords (Tensor): A tensor of shape (R, P, 2) that contains
            image-normalized coordinates of P sampled points.
    Returns:
        point_coords_wrt_box (Tensor): A tensor of shape (R, P, 2) that contains
            [0, 1] x [0, 1] box-normalized coordinates of the P sampled points.
    """
    with torch.no_grad():
        point_coords_wrt_box = point_coords.clone()
        point_coords_wrt_box[:, :, 0] -= boxes_coords[:, None, 0]
        point_coords_wrt_box[:, :, 1] -= boxes_coords[:, None, 1]
        point_coords_wrt_box[:, :, 0] = point_coords_wrt_box[:, :, 0] / (
            boxes_coords[:, None, 2] - boxes_coords[:, None, 0]
        )
        point_coords_wrt_box[:, :, 1] = point_coords_wrt_box[:, :, 1] / (
            boxes_coords[:, None, 3] - boxes_coords[:, None, 1]
        )
    return point_coords_wrt_box


def unfold_wo_center(x, kernel_size, dilation):
    assert x.dim() == 4
    assert kernel_size % 2 == 1

    # using SAME padding
    padding = (kernel_size + (dilation - 1) * (kernel_size - 1)) // 2
    unfolded_x = F.unfold(
        x, kernel_size=kernel_size,
        padding=padding,
        dilation=dilation
    )

    unfolded_x = unfolded_x.reshape(
        x.size(0), x.size(1), -1, x.size(2), x.size(3)
    )

    # remove the center pixels
    size = kernel_size ** 2
    unfolded_x = torch.cat((
        unfolded_x[:, :, :size // 2],
        unfolded_x[:, :, size // 2 + 1:]
    ), dim=2)

    return unfolded_x


def get_images_color_similarity(images, kernel_size, dilation):
    assert images.dim() == 4
    assert images.size(0) == 1

    unfolded_images = unfold_wo_center(
        images, kernel_size=kernel_size, dilation=dilation
    )

    diff = images[:, :, None] - unfolded_images
    similarity = torch.exp(-torch.norm(diff, dim=1) * 0.5)

    # unfolded_weights = unfold_wo_center(
    #     image_masks[None, None], kernel_size=kernel_size,
    #     dilation=dilation
    # )
    # unfolded_weights = torch.max(unfolded_weights, dim=1)[0]

    return similarity #* unfolded_weights


def add_bitmasks_from_boxes(instances, images, image_masks, im_h, im_w, mask_out_stride, pairwise_size, pairwise_dilation):
    stride = mask_out_stride
    start = int(stride // 2)

    assert images.size(2) % stride == 0
    assert images.size(3) % stride == 0

    downsampled_images = F.avg_pool2d(
        images.float(), kernel_size=stride,
        stride=stride, padding=0
    )[:, [2, 1, 0]]
    image_masks = image_masks[:, start::stride, start::stride]

    for im_i, per_im_gt_inst in enumerate(instances):
        images_lab = color.rgb2lab(downsampled_images[im_i].byte().permute(1, 2, 0).cpu().numpy())
        images_lab = torch.as_tensor(images_lab, device=downsampled_images.device, dtype=torch.float32)
        images_lab = images_lab.permute(2, 0, 1)[None]
        images_color_similarity = get_images_color_similarity(
            images_lab, pairwise_size, pairwise_dilation
        )

        per_im_gt_inst.image_color_similarity = torch.cat([
            images_color_similarity for _ in range(len(per_im_gt_inst))
        ], dim=0)

# def compute_project_term(mask_scores, gt_bitmasks):
#     mask_losses_y = dice_coefficient(
#         mask_scores.max(dim=2, keepdim=True)[0],
#         gt_bitmasks.max(dim=2, keepdim=True)[0]
#     )
#     mask_losses_x = dice_coefficient(
#         mask_scores.max(dim=3, keepdim=True)[0],
#         gt_bitmasks.max(dim=3, keepdim=True)[0]
#     )
#     return (mask_losses_x + mask_losses_y).mean()


def compute_pairwise_term(mask_logits, pairwise_size, pairwise_dilation):
    assert mask_logits.dim() == 4

    log_fg_prob = F.logsigmoid(mask_logits)
    log_bg_prob = F.logsigmoid(-mask_logits)

    log_fg_prob_unfold = unfold_wo_center(
        log_fg_prob, kernel_size=pairwise_size,
        dilation=pairwise_dilation
    )
    log_bg_prob_unfold = unfold_wo_center(
        log_bg_prob, kernel_size=pairwise_size,
        dilation=pairwise_dilation
    )

    # the probability of making the same prediction = p_i * p_j + (1 - p_i) * (1 - p_j)
    # we compute the the probability in log space to avoid numerical instability
    log_same_fg_prob = log_fg_prob[:, :, None] + log_fg_prob_unfold
    log_same_bg_prob = log_bg_prob[:, :, None] + log_bg_prob_unfold

    max_ = torch.max(log_same_fg_prob, log_same_bg_prob)
    log_same_prob = torch.log(
        torch.exp(log_same_fg_prob - max_) +
        torch.exp(log_same_bg_prob - max_)
    ) + max_

    # loss = -log(prob)
    return -log_same_prob[:, 0]