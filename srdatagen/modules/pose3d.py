import math
from typing import Any
from typing import Dict
from typing import Tuple

import numpy as np
from PIL import Image
import torch
from transformers import AutoImageProcessor

from srdatagen import AnnotType
from srdatagen.pose_utils import bin_to_continuous, PoseDINOv2


class Pose3D:
    """Predict 3D poses for objects in the image.
    """

    def __init__(self, cfg, device):
        self.cfg = cfg
        self.device = device

        self.pose_model = PoseDINOv2(
            backbone=self.cfg.pose.backbone,
            heads=self.cfg.pose.heads,
            layers='4_uniform',
            avgpool=False)
        state = torch.load(self.cfg.pose.ckpt, map_location='cpu')
        filtered_state = {}
        for k in state:
            if k.startswith('module.'):
                filtered_state[k[7:]] = state[k]
            else:
                filtered_state[k] = state[k]
        self.pose_model.load_state_dict(filtered_state)
        self.pose_model.eval().to(device)

        self.image_processor = AutoImageProcessor.from_pretrained(
            pretrained_model_name_or_path=self.cfg.pose.backbone)

        self.G = np.eye(3)
        self.G[0, 0] = -1.0
        self.G[1, 1] = -1.0

    def __call__(self, image: Image.Image, annot: AnnotType, debug_prompt: str = None) -> AnnotType:
        for idx, obj in enumerate(annot['detections']):
            if not obj['class_name'] in self.cfg.pose.class_names:
                continue
            if 'pcd_center' not in obj:
                continue
            azim, elev, theta = self._predict_pose(image, annot, obj, debug_prompt=f'pose_{idx:02d}' if debug_prompt else None)
            left, front, up = self._compute_directions(azim, elev, theta, obj['pcd_center'], annot)
            obj['pose'] = [azim, elev, theta]
            obj['left'], obj['front'], obj['up'] = left, front, up
        return annot

    def _predict_pose(self, image: Image.Image, annot: AnnotType, obj: Dict[str, Any], debug_prompt: str = None) -> Dict[str, float]:
        img = np.array(image)
        xmin, ymin, xmax, ymax = [int(np.rint(x)) for x in obj['xyxy']]
        width = xmax - xmin + 1
        height = ymax - ymin + 1
        if width > height:
            pady = (width - height) // 2
            ymin -= pady
            ymax += width - height - pady
        elif width < height:
            padx = (height - width) // 2
            xmin -= padx
            xmax += height - width - padx
        xmid = (xmin + xmax) // 2
        ymid = (ymin + ymax) // 2
        xmin1 = int(xmid - (xmax - xmin) / 2 * self.cfg.pose.padding)
        xmax1 = int(xmid + (xmax - xmin) / 2 * self.cfg.pose.padding)
        ymin1 = int(ymid - (ymax - ymin) / 2 * self.cfg.pose.padding)
        ymax1 = int(ymid + (ymax - ymin) / 2 * self.cfg.pose.padding)
        p = max(xmax - xmin, ymax - ymin)
        img = np.pad(img, ((p, p), (p, p), (0, 0)))
        xmin1 += p
        xmax1 += p
        ymin1 += p
        ymax1 += p
        img = img[ymin1:ymax1+1, xmin1:xmax1+1]

        if debug_prompt:
            annot['vis'][debug_prompt] = Image.fromarray(img)

        img_tensor = self.image_processor(
            Image.fromarray(img), return_tensors='pt').pixel_values.to(self.device)
        azim, elev, theta = self.pose_model(img_tensor)

        azimuth_pred = bin_to_continuous(np.argmax(azim.detach().cpu().numpy(), axis=-1), **self.cfg.pose.multi_bin)
        elevation_pred = bin_to_continuous(np.argmax(elev.detach().cpu().numpy(), axis=-1), **self.cfg.pose.multi_bin)
        theta_pred = bin_to_continuous(np.argmax(theta.detach().cpu().numpy(), axis=-1), **self.cfg.pose.multi_bin)

        # print(f'azim={float(azimuth_pred)/np.pi*180:.1f}, elev={float(elevation_pred)/np.pi*180:.1f}, theta={float(theta_pred)/np.pi*180:.1f}')

        return float(azimuth_pred[0]), float(elevation_pred[0]), float(theta_pred[0])

    def _compute_directions(self, azim: float, elev: float, theta: float, T: np.ndarray, annot: AnnotType) -> Tuple[np.ndarray]:
        # Default axis centered at (0, 0, 0)
        verts = np.array([
            [0.0, 0.0, 0.0],
            [-1.0, 0.0, 0.0],
            [0.0, 0.0, -1.0],
            [0.0, 1.0, 0.0]])

        # Rotate default axis by azim, elev, theta
        Ry = np.array([
            [math.cos(-azim), 0.0, math.sin(-azim)],
            [0.0, 1.0, 0.0],
            [-math.sin(-azim), 0.0, math.cos(-azim)]])
        Rx = np.array([
            [1.0, 0.0, 0.0],
            [0.0, math.cos(-elev), -math.sin(-elev)],
            [0.0, math.sin(-elev), math.cos(-elev)]])
        Rz = np.array([
            [math.cos(-theta), -math.sin(-theta), 0.0],
            [math.sin(-theta), math.cos(-theta), 0.0],
            [0.0, 0.0, 1.0]])
        R = np.dot(Rz, np.dot(Rx, Ry))
        verts = np.dot(R, verts.T).T

        # Move axis to [0, 0, distance]
        verts += np.array([0.0, 0.0, np.linalg.norm(T)])

        # Rotate axis to align with T
        p_orig = np.array([0.0, 0.0, 1.0])
        p_new = T / np.linalg.norm(T)
        axis = np.cross(p_orig, p_new)
        ux, uy, uz = axis / np.linalg.norm(axis)
        cos_theta = np.dot(p_orig, p_new)
        theta = np.arccos(cos_theta)
        K = np.array([
            [0, -uz, uy],
            [uz, 0, -ux],
            [-uy, ux, 0]])
        I = np.eye(3)
        R1 = I + np.sin(theta) * K + (1 - np.cos(theta)) * np.dot(K, K)
        verts = np.dot(R1, verts.T).T

        # Move axis to canonical 3D space
        verts = verts @ self.G @ annot['scene_3d_info']['pf_R'] @ self.G.T - annot['scene_3d_info']['min_y']

        # Normalize axis
        verts = verts[1:] - verts[0:1]
        norms = np.linalg.norm(verts, axis=1, keepdims=True)
        verts = verts / norms
        return verts[0], verts[1], verts[2]
