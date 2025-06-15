from datetime import datetime
from itertools import groupby
import logging
import os
from typing import Any, Dict

import numpy as np

AnnotType = Dict[str, Any]

COLORS_8 = [
    [119, 170, 221],
    [153, 221, 255],
    [68, 187, 153],
    [187, 204, 51],
    [170, 170, 0],
    [238, 221, 136],
    [238, 136, 102],
    [255, 170, 187]]
COLORS_8_F = np.array(COLORS_8) / 255.0


class SkipSampleException(Exception):
    """Raise this exception if current sample should be skipped for various
    reasons.
    """
    pass


def setup_logging(save_path=None):
    if save_path is None:
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S')
    else:
        os.path.makedirs(save_path, exist_ok=True)
        dt = datetime.now().strftime('%Y%m%d_%H%M%S')
        logging.root.handlers = []
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            filename=os.path.join(save_path, f'log_{dt}.txt'),
            filemode='w',
        )
        console = logging.StreamHandler()
        console.setLevel(logging.INFO)
        formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
        console.setFormatter(formatter)
        logging.getLogger("").addHandler(console)
        return logging.getLogger("").handlers[0].baseFilename


def serialize_pcd(pcd, save_pcd=False):
    pcd_points = np.asarray(pcd.points)
    size1 = np.max(pcd_points[:, 0]) - np.min(pcd_points[:, 0])
    size2 = np.max(pcd_points[:, 1]) - np.min(pcd_points[:, 1])
    size3 = np.max(pcd_points[:, 2]) - np.min(pcd_points[:, 2])
    if save_pcd:
        return dict(
            type='open3d.cuda.pybind.geometry.PointCloud',
            points=np.asarray(pcd.points).tolist(),
            colors=np.asarray(pcd.colors).tolist(),
            size=[size1, size2, size3])
    else:
        return dict(
            type='open3d.cuda.pybind.geometry.PointCloud',
            size=[size1, size2, size3])


def mask_to_rle(binary_mask):
    rle = {'counts': [], 'size': list(binary_mask.shape)}
    counts = rle.get('counts')
    for i, (value, elements) in enumerate(groupby(binary_mask.ravel(order='F'))):
        if i == 0 and value == 1:
            counts.append(0)
        counts.append(len(list(elements)))
    return rle


def serialize(annot: AnnotType, save_pcd: bool = False) -> AnnotType:
    if 'vis' in annot:
        annot.pop('vis')
    for k in annot['scene_3d_info']:
        if k in ['intrinsic', 'pf_R', 'min_y']:
            annot['scene_3d_info'][k] = annot['scene_3d_info'][k].tolist()
        elif k in ['pcd', 'pcd_cano']:
            # annot['scene_3d_info'][k] = dict(
            #     type='open3d.cuda.pybind.geometry.PointCloud',
            #     points=np.asarray(annot['scene_3d_info'][k].points).tolist(),
            #     colors=np.asarray(annot['scene_3d_info'][k].colors).tolist() if annot['scene_3d_info'][k].has_colors() else None,
            #     normals=np.asarray(annot['scene_3d_info'][k].normals).tolist() if annot['scene_3d_info'][k].has_normals() else None)
            annot['scene_3d_info'][k] = serialize_pcd(annot['scene_3d_info'][k], save_pcd=save_pcd)
        else:
            raise NotImplementedError(f'Unknown key {k} with dtype {type(annot["scene_3d_info"][k])}')
    for obj in annot['detections']:
        for k in list(obj.keys()):
            if k in ['class_name', 'confidence', 'box_area', 'area', 'class_id', 'pose', 'object_name']:
                pass
            elif k in ['xyxy', 'pcd_center', 'pcd_cano_center', 'left', 'front', 'up']:
                obj[k] = obj[k].tolist()
            elif k in ['mask', 'mask_subtracted']:
                obj[k+'_rle'] = mask_to_rle(obj[k])
                obj.pop(k)
            elif k in ['pcd', 'pcd_cano']:
                # obj[k] = dict(
                #     type='open3d.cuda.pybind.geometry.PointCloud',
                #     points=np.asarray(obj[k].points).tolist(),
                #     colors=np.asarray(obj[k].colors).tolist() if obj[k].has_colors() else None,
                #     normals=np.asarray(obj[k].normals).tolist() if obj[k].has_normals() else None)
                obj[k] = serialize_pcd(obj[k], save_pcd=save_pcd)
            elif k in ['pcd_axis_bbox', 'pcd_cano_axis_bbox', 'pcd_orient_bbox', 'pcd_cano_orient_bbox']:
                obj[k] = {_k: obj[k][_k].tolist() for _k in obj[k]}
            else:
                raise NotImplementedError(f'Unknown key {k} with dtype {type(obj[k])}')
    return annot