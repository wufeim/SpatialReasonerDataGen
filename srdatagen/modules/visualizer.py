import os
from typing import Tuple

import cv2
import numpy as np
import open3d as o3d
from PIL import Image
from scipy.spatial.transform import Rotation
from transforms3d import affines, euler
# from wis3d import Wis3D

from srdatagen.utils import AnnotType
from srdatagen.utils import COLORS_8_F
from srdatagen.utils import draw_text
from srdatagen.utils import VisImage

EDGES = [(0, 1), (1, 2), (2, 3), (3, 0), (4, 5), (5, 6), (6, 7), (7, 4), (0, 4), (1, 5), (2, 6), (3, 7)]


def project_3d_to_2d(pts_3d, intrinsic, h, w):
    pts_2d = np.dot(intrinsic, pts_3d.T)
    pts_2d = pts_2d[:2, :] / pts_2d[2, :]
    pts_2d = pts_2d.T
    pts_2d[:, 0] = pts_2d[:, 0]
    pts_2d[:, 1] = pts_2d[:, 1]
    pts_2d[:, 0] = w - pts_2d[:, 0]
    pts_2d[:, 1] = h - pts_2d[:, 1]
    return pts_2d


class Visualizer:
    """Visualizing 3D reconstruction results."""

    def __init__(self, cfg):
        self.cfg = cfg

        self.G = np.eye(3)
        self.G[0, 0] = -1.0
        self.G[1, 1] = -1.0

    def __call__(self, image: Image.Image, annot: AnnotType, output_path: str) -> Image.Image:
        os.makedirs(output_path, exist_ok=True)
        # wis3d = Wis3D(output_path, output_filename)

        # # Add scene pcd
        # pts3d, pts3d_color = self._pcd_to_pts3d_and_colors(annot['scene_3d_info']['pcd_cano'])
        # wis3d.add_point_cloud(vertices=pts3d, colors=pts3d_color, name='scene_pcd_cano')

        # # Add obj pcd
        # for obj_idx, obj in enumerate(annot['detections']):
        #     pts3d, pts3d_color = self._pcd_to_pts3d_and_colors(obj['pcd_cano'])
        #     wis3d.add_point_cloud(vertices=pts3d, colors=pts3d_color, name=f'{obj_idx:02d}_{obj["class_name"]}')
        #     bbox = obj['pcd_cano_axis_bbox']
        #     wis3d.add_boxes(positions=bbox['center'], eulers=bbox['eulers'], extents=bbox['extent'], name=f'{obj_idx:02d}_{obj["class_name"]}_bbox')

        # # Add obj axis
        # for obj_idx, obj in enumerate(annot['detections']):
        #     if 'left' not in obj:
        #         continue
        #     rays_o = np.array([
        #         obj['pcd_cano_center'],
        #         obj['pcd_cano_center'],
        #         obj['pcd_cano_center']])
        #     rays_d = np.array([
        #         obj['left'],
        #         obj['front'],
        #         obj['up']])
        #     wis3d.add_rays(rays_o, rays_d, name=f'{obj_idx:02d}_{obj["class_name"]}_axis')

        img = np.array(image)
        pf_R = np.asarray(annot['scene_3d_info']['pf_R'])
        intrinsic = np.asarray(annot['scene_3d_info']['intrinsic'])
        min_y = np.asarray(annot['scene_3d_info']['min_y'])
        for obj_idx, obj in enumerate(annot['detections']):
            if 'pose' not in obj:
                continue

            # Draw 2D bbox
            xyxy = obj['xyxy']
            img = cv2.rectangle(img, (int(xyxy[0]), int(xyxy[1])), (int(xyxy[2]), int(xyxy[3])), (0, 255, 0), 2)

            # Draw 3D pcd
            if 'points' in obj['pcd_cano']:
                pts3d = np.asarray(obj['pcd_cano']['points'])
                pts3d_uncano = self._uncanonicalize(pts3d, pf_R, min_y)
                # pts3d_uncano = np.asarray(obj['pcd']['points'])
                pts2d = project_3d_to_2d(pts3d_uncano, intrinsic, intrinsic[1, 2]*2, intrinsic[0, 2]*2)
                for i in range(pts2d.shape[0]):
                    img = cv2.circle(img, (int(pts2d[i, 0]), int(pts2d[i, 1])), 1, (255, 0, 0), -1)

            # Draw 3D center
            pts3d = self._uncanonicalize(np.asarray(obj['pcd_cano_center'])[None, :], pf_R, min_y)
            pts2d = project_3d_to_2d(pts3d, intrinsic, intrinsic[1, 2]*2, intrinsic[0, 2]*2)
            img = cv2.circle(img, (int(pts2d[0, 0]), int(pts2d[0, 1])), 3, (0, 255, 0), -1)

            # Draw axis
            axis3d = np.array([
                np.asarray(obj['pcd_cano_center']),
                np.asarray(obj['pcd_cano_center']) + np.asarray(obj['left']),
                np.asarray(obj['pcd_cano_center']) + np.asarray(obj['front']),
                np.asarray(obj['pcd_cano_center']) + np.asarray(obj['up'])])
            axis3d = self._uncanonicalize(axis3d, pf_R, min_y)
            axis2d = project_3d_to_2d(axis3d, intrinsic, intrinsic[1, 2]*2, intrinsic[0, 2]*2)
            img = cv2.circle(img, (int(pts2d[0, 0]), int(pts2d[0, 1])), 3, (0, 255, 0), -1)
            img = cv2.arrowedLine(img, (int(axis2d[0, 0]), int(axis2d[0, 1])), (int(axis2d[1, 0]), int(axis2d[1, 1])), (0, 255, 0), 2)
            img = cv2.arrowedLine(img, (int(axis2d[0, 0]), int(axis2d[0, 1])), (int(axis2d[2, 0]), int(axis2d[2, 1])), (255, 0, 0), 2)
            img = cv2.arrowedLine(img, (int(axis2d[0, 0]), int(axis2d[0, 1])), (int(axis2d[3, 0]), int(axis2d[3, 1])), (0, 0, 255), 2)

            # Draw pose
            img = draw_text(
                VisImage(img), axis2d[0, 0], axis2d[0, 1],
                text=str([f'{round(np.degrees(x))}' for x in obj['pose']]),
                color=COLORS_8_F[obj_idx % 8]
            ).get_image()

            # Draw oriented bbox
            # bbox = obj['pcd_cano_axis_bbox']
            # center, eulers, extent = bbox['center'], bbox['eulers'], bbox['extent']
            # ex, ey, ez = extent
            # bbox = np.array([
            #     [ex, ey, ez], [ex, -ey, ez], [-ex, -ey, ez], [-ex, ey, ez],
            #     [ex, ey, -ez], [ex, -ey, -ez], [-ex, -ey, -ez], [-ex, ey, -ez]])
            # rotation_matrix = Rotation.from_euler('XYZ', eulers).as_matrix()
            # bbox = np.dot(bbox, rotation_matrix.T) + center
            # axis3d = self._uncanonicalize(bbox, pf_R, min_y)
            # axis2d = project_3d_to_2d(axis3d, intrinsic, img.shape[0], img.shape[1])
            # for ei, ej in EDGES:
            #     img = cv2.line(img, (int(axis2d[ei, 0]), int(axis2d[ei, 1])), (int(axis2d[ej, 0]), int(axis2d[ej, 1])), (255, 255, 0), 2)

        Image.fromarray(img).save(os.path.join(output_path, 'visualization.png'))

    def _uncanonicalize(self, pts3d: np.ndarray, R: np.ndarray, min_y: np.ndarray) -> np.ndarray:
        return (pts3d+min_y).reshape((-1, 3)) @ self.G @ np.linalg.inv(R) @ self.G.T

    def _pcd_to_pts3d_and_colors(self, pcd: o3d.geometry.PointCloud) -> Tuple[np.ndarray, np.ndarray]:
        pts3d = np.asarray(pcd.points)
        if pcd.has_colors():
            colors = np.asarray(pcd.colors)
        else:
            colors = np.zeros_like(pts3d)
        return pts3d, colors
