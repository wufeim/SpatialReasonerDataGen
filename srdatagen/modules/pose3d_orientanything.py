import math
from typing import Any
from typing import Dict
from typing import Tuple

import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F
from transformers import AutoImageProcessor

from srdatagen.utils import AnnotType
from srdatagen.orientanything_utils import DINOv2_MLP


def get_3angle(image, dino, val_preprocess, device):
    image_inputs = val_preprocess(images=image)
    image_inputs['pixel_values'] = torch.from_numpy(np.array(image_inputs['pixel_values'])).to(device)
    with torch.no_grad():
        dino_pred = dino(image_inputs)
    gaus_ax_pred   = torch.argmax(dino_pred[:, 0:360], dim=-1)
    gaus_pl_pred   = torch.argmax(dino_pred[:, 360:360+180], dim=-1)
    gaus_ro_pred   = torch.argmax(dino_pred[:, 360+180:360+180+180], dim=-1)
    confidence     = F.softmax(dino_pred[:, -2:], dim=-1)[0][0]
    angles = torch.zeros(4)
    angles[0]  = gaus_ax_pred
    angles[1]  = gaus_pl_pred - 90
    angles[2]  = gaus_ro_pred - 90
    angles[3]  = confidence
    return angles


class Pose3DOrientAnything:
    """Predict 3D poses for objects in the image.
    """

    def __init__(self, cfg, device):
        self.cfg = cfg
        self.device = device

        self.pose_model = DINOv2_MLP(
            dino_mode=self.cfg.orientanything.dino_mode,
            in_dim=self.cfg.orientanything.in_dim,
            out_dim=self.cfg.orientanything.out_dim,
            evaluate=True,
            mask_dino=False,
            frozen_back=False)
        self.pose_model.load_state_dict(torch.load(cfg.orientanything.ckpt_path, map_location='cpu'))
        self.pose_model.eval().to(device)

        self.image_processor = AutoImageProcessor.from_pretrained(
            pretrained_model_name_or_path=self.cfg.orientanything.backbone)

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

        if 'vis' not in annot:
            annot['vis'] = {}
        if debug_prompt:
            annot['vis'][debug_prompt] = Image.fromarray(img)

        azimuth_pred, elevation_pred, theta_pred, confidence = get_3angle(
            Image.fromarray(img), self.pose_model,
            self.image_processor, self.device)
        azimuth_pred = azimuth_pred / 180 * np.pi
        elevation_pred = elevation_pred / 180 * np.pi
        theta_pred = theta_pred / 180 * np.pi

        return float(azimuth_pred), float(elevation_pred), float(theta_pred)

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
