
def add_point_sup_config(cfg):
    """
    Add config for point supervision.
    """
    # Use point annotation
    cfg.INPUT.POINT_SUP = False
    # Sample only part of points in each iteration.
    # Default: 0, use all available points.
    cfg.INPUT.SAMPLE_POINTS = 0
    
    # using the pixel similarity loss to semi-supervise the learning
    cfg.INPUT.PIXEL_SIMILARITY = False

    # Values to be used for image normalization (BGR order, since INPUT.FORMAT defaults to BGR).
    # To train on images of different number of channels, just set different mean & std.
    # Default values are the mean pixel value from ImageNet: [103.53, 116.28, 123.675]
    cfg.MODEL.PIXEL_MEAN = [103.530, 116.280, 123.675]#[123.675, 116.28, 103.53] #
    # When using pre-trained models in Detectron1 or any MSRA models,
    # std has been absorbed into its conv1 weights, so the std needs to be set 1.
    # Otherwise, you can use [57.375, 57.120, 58.395] (ImageNet std)
    cfg.MODEL.PIXEL_STD = [1.0, 1.0, 1.0]#[58.395, 57.12, 57.375]#