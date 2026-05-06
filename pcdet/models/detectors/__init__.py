from importlib import import_module


DETECTOR_REGISTRY = {
    'Detector3DTemplate': ('detector3d_template', 'Detector3DTemplate'),
    'SECONDNet': ('second_net', 'SECONDNet'),
    'PartA2Net': ('PartA2_net', 'PartA2Net'),
    'PVRCNN': ('pv_rcnn', 'PVRCNN'),
    'PointPillar': ('pointpillar', 'PointPillar'),
    'PointRCNN': ('point_rcnn', 'PointRCNN'),
    'SECONDNetIoU': ('second_net_iou', 'SECONDNetIoU'),
    'CaDDN': ('caddn', 'CaDDN'),
    'VoxelRCNN': ('voxel_rcnn', 'VoxelRCNN'),
    'CenterPoint': ('centerpoint', 'CenterPoint'),
    'PVRCNNPlusPlus': ('pv_rcnn_plusplus', 'PVRCNNPlusPlus'),
    'Detector3DTemplate_voxel_mae': ('detector3d_template_voxel_mae', 'Detector3DTemplate_voxel_mae'),
    'Voxel_MAE': ('voxel_mae_net', 'Voxel_MAE'),
    'AD_L_JEPA': ('ad_l_jepa_net', 'AD_L_JEPA'),
}


def build_detector(model_cfg, num_class, dataset):
    if model_cfg.NAME not in DETECTOR_REGISTRY:
        available = ', '.join(sorted(DETECTOR_REGISTRY))
        raise KeyError(f'Unknown detector {model_cfg.NAME}. Available detectors: {available}')

    module_name, class_name = DETECTOR_REGISTRY[model_cfg.NAME]
    detector_cls = getattr(import_module(f'{__name__}.{module_name}'), class_name)
    model = detector_cls(
        model_cfg=model_cfg, num_class=num_class, dataset=dataset
    )

    return model
