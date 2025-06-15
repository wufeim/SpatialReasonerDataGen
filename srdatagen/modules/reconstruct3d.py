from collections import Counter
import sys
from typing import Dict

import cv2
import numpy as np
import open3d as o3d
from PIL import Image
from scipy.spatial.transform import Rotation
import torch

sys.path.append('PerspectiveFields')
from perspective2d import PerspectiveFields
from perspective2d.utils import draw_from_r_p_f_cx_cy

sys.path.append('Depth-Anything-V2/metric_depth')
from depth_anything_v2.dpt import DepthAnythingV2

from srdatagen import AnnotType

DAv2_configs = {
    'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]},
    'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]},
    'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]}
}


def pcd_denoise_dbscan(pcd: o3d.geometry.PointCloud, eps=0.02, min_points=10) -> o3d.geometry.PointCloud:
    # Remove noise via clustering
    pcd_clusters = pcd.cluster_dbscan(
        eps=eps,
        min_points=min_points,
    )

    # Convert to numpy arrays
    obj_points = np.asarray(pcd.points)
    obj_colors = np.asarray(pcd.colors)
    pcd_clusters = np.array(pcd_clusters)

    # Count all labels in the cluster
    counter = Counter(pcd_clusters)

    # Remove the noise label
    if counter and (-1 in counter):
        del counter[-1]

    if counter:
        # Find the label of the largest cluster
        most_common_label, _ = counter.most_common(1)[0]

        # Create mask for points in the largest cluster
        largest_mask = pcd_clusters == most_common_label

        # Apply mask
        largest_cluster_points = obj_points[largest_mask]
        largest_cluster_colors = obj_colors[largest_mask]

        # If the largest cluster is too small, return the original point cloud
        if len(largest_cluster_points) < 5:
            return pcd

        # Create a new PointCloud object
        largest_cluster_pcd = o3d.geometry.PointCloud()
        largest_cluster_pcd.points = o3d.utility.Vector3dVector(largest_cluster_points)
        largest_cluster_pcd.colors = o3d.utility.Vector3dVector(largest_cluster_colors)

        pcd = largest_cluster_pcd

    return pcd


def resize_fix_aspect_ratio(img, field, target_width=None, target_height=None):
    height = img.shape[0]
    width = img.shape[1]
    if target_height is None:
        factor = target_width / width
    elif target_width is None:
        factor = target_height / height
    else:
        factor = max(target_width / width, target_height / height)
    if factor == target_width / width:
        target_height = int(height * factor)
    else:
        target_width = int(width * factor)

    img = cv2.resize(img, (target_width, target_height))
    for key in field:
        if key not in ["up", "lati"]:
            continue
        tmp = field[key].numpy()
        transpose = len(tmp.shape) == 3
        if transpose:
            tmp = tmp.transpose(1, 2, 0)
        tmp = cv2.resize(tmp, (target_width, target_height))
        if transpose:
            tmp = tmp.transpose(2, 0, 1)
        field[key] = torch.tensor(tmp)
    return img, field


class Reconstruct3D:
    """Reconstruct the scene in 3D."""

    def __init__(self, cfg, device):
        self.cfg = cfg
        self.device = device

        self.perspective_fields_model = PerspectiveFields(
            self.cfg.reconstruct3d.perspective_fields_model_name).eval().to(self.device)
        self.wild_camera_model = torch.hub.load(
            self.cfg.reconstruct3d.wild_camera_model, 'WildCamera', pretrained=True
        ).to(self.device)

        # self.depth_model = DepthAnythingV2(
        #     **DAv2_configs[self.cfg.reconstruct3d.dav2_backbone], max_depth=self.cfg.reconstruct3d.max_depth)
        # self.depth_model.load_state_dict(torch.load(
        #     self.cfg.reconstruct3d.dav2_ckpt_path, map_location='cpu'))
        # self.depth_model.eval().to(self.device)

        self.depth_model = torch.hub.load("yvanyin/metric3d", "metric3d_vit_giant2", pretrain=True)
        self.depth_model.eval().to(self.device)

    @torch.no_grad()
    def __call__(self, image: Image.Image, annot: AnnotType, debug_prompt: str = None, run_filter: bool = True) -> AnnotType:
        pf_params = self._run_perspective_fields(image, annot, debug_prompt=debug_prompt)
        pf_R = self._create_rotation_matrix(
            roll=pf_params['roll'], pitch=pf_params['pitch'], yaw=0.0)
        intrinsic = self._run_wild_camera(image)
        depth = self._run_depth(image, intrinsic)

        pts3d = self._depth_to_points(depth[None], intrinsic=intrinsic)
        pts3d_cano = self._depth_to_points(depth[None], R=pf_R.T, intrinsic=intrinsic)
        pts3d_cano, min_y = self._move_to_ground(pts3d_cano)

        for obj_idx, obj in enumerate(annot['detections']):
            mask = obj['mask_subtracted']

            object_pcd = self._create_object_pcd(pts3d, np.array(image), mask)
            object_pcd_cano = self._create_object_pcd(pts3d_cano, np.array(image), mask)
            if run_filter and (
                len(object_pcd.points) < self.cfg.reconstruct3d.min_points_threshold or
                len(object_pcd_cano.points) < self.cfg.reconstruct3d.min_points_threshold
            ):
                continue

            object_pcd = self._process_pcd(object_pcd)
            object_pcd_cano = self._process_pcd(object_pcd_cano)
            if run_filter and (
                len(object_pcd.points) < self.cfg.reconstruct3d.min_points_threshold_dbscan or
                len(object_pcd_cano.points) < self.cfg.reconstruct3d.min_points_threshold_dbscan
            ):
                continue

            obj['pcd'] = object_pcd
            obj['pcd_center'] = object_pcd.get_center()
            obj['pcd_cano'] = object_pcd_cano
            obj['pcd_cano_center'] = object_pcd_cano.get_center()

            obj['pcd_axis_bbox'] = self._get_axis_aligned_bounding_box(object_pcd)
            obj['pcd_cano_axis_bbox'] = self._get_axis_aligned_bounding_box(object_pcd_cano)
            obj['pcd_orient_bbox'] = self._get_oriented_bounding_box(object_pcd)
            obj['pcd_cano_orient_bbox'] = self._get_oriented_bounding_box(object_pcd_cano)

        scene_pcd = self._create_object_pcd(pts3d, np.array(image))
        scene_pcd_cano = self._create_object_pcd(pts3d_cano, np.array(image))
        annot['scene_3d_info'] = dict(
            intrinsic=intrinsic, pf_R=pf_R, pcd=scene_pcd,
            pcd_cano=scene_pcd_cano, min_y=min_y)

        return annot

    def _run_perspective_fields(self, image: Image.Image, annot: AnnotType, debug_prompt: str = None) -> Dict[str, float]:
        model_input = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        pred = self.perspective_fields_model.inference(img_bgr=model_input)

        return dict(
            roll=pred['pred_roll'].cpu().item(),
            pitch=pred['pred_pitch'].cpu().item())

    def _create_rotation_matrix(self, roll: float, pitch: float, yaw: float) -> np.ndarray:
        roll = np.radians(roll)
        pitch = np.radians(pitch)
        yaw = np.radians(yaw)

        R_x = np.array([
            [1.0, 0.0, 0.0],
            [0.0, np.cos(pitch), np.sin(pitch)],
            [0.0, -np.sin(pitch), np.cos(pitch)]])
        R_y = np.array([
            [np.cos(yaw), 0.0, -np.sin(yaw)],
            [0.0, 1.0, 0.0],
            [np.sin(yaw), 0.0, np.cos(yaw)]])
        R_z = np.array([
            [np.cos(roll), np.sin(roll), 0.0],
            [-np.sin(roll), np.cos(roll), 0.0],
            [0.0, 0.0, 1.0]])

        return R_z @ R_x @ R_y

    def _run_wild_camera(self, image: Image.Image) -> np.ndarray:
        intrinsic, _ = self.wild_camera_model.inference(image, wtassumption=False)
        return intrinsic

    def _run_depth(self, image: Image.Image, intrinsic: np.ndarray) -> np.ndarray:
        # Code from # https://github.com/YvanYin/Metric3D/blob/main/hubconf.py, assume rgb_origin is in RGB
        intrinsic = [intrinsic[0, 0], intrinsic[1, 1], intrinsic[0, 2], intrinsic[1, 2]]

        rgb_origin = np.array(image)

        # ajust input size to fit pretrained model
        # keep ratio resize
        input_size = (616, 1064)  # for vit model
        # input_size = (544, 1216) # for convnext model
        h, w = rgb_origin.shape[:2]
        scale = min(input_size[0] / h, input_size[1] / w)
        rgb = cv2.resize(rgb_origin, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_LINEAR)
        # remember to scale intrinsic, hold depth
        intrinsic = [intrinsic[0] * scale, intrinsic[1] * scale, intrinsic[2] * scale, intrinsic[3] * scale]

        # padding to input_size
        padding = [123.675, 116.28, 103.53]
        h, w = rgb.shape[:2]
        pad_h = input_size[0] - h
        pad_w = input_size[1] - w
        pad_h_half = pad_h // 2
        pad_w_half = pad_w // 2
        rgb = cv2.copyMakeBorder(
            rgb, pad_h_half, pad_h - pad_h_half, pad_w_half, pad_w - pad_w_half, cv2.BORDER_CONSTANT, value=padding
        )
        pad_info = [pad_h_half, pad_h - pad_h_half, pad_w_half, pad_w - pad_w_half]

        # normalize
        mean = torch.tensor([123.675, 116.28, 103.53]).float()[:, None, None]
        std = torch.tensor([58.395, 57.12, 57.375]).float()[:, None, None]
        rgb = torch.from_numpy(rgb.transpose((2, 0, 1))).float()
        rgb = torch.div((rgb - mean), std)
        rgb = rgb[None, :, :, :].cuda()

        with torch.no_grad():
            pred_depth, confidence, output_dict = self.depth_model.inference({"input": rgb})

        # un pad
        pred_depth = pred_depth.squeeze()
        pred_depth = pred_depth[
            pad_info[0] : pred_depth.shape[0] - pad_info[1], pad_info[2] : pred_depth.shape[1] - pad_info[3]
        ]

        # upsample to original size
        pred_depth = torch.nn.functional.interpolate(
            pred_depth[None, None, :, :], rgb_origin.shape[:2], mode="bilinear"
        ).squeeze()

        # de-canonical transform
        canonical_to_real_scale = intrinsic[0] / 1000.0  # 1000.0 is the focal length of canonical camera
        pred_depth = pred_depth * canonical_to_real_scale  # now the depth is metric
        pred_depth = torch.clamp(pred_depth, 0, 300)
        return pred_depth.detach().cpu().numpy()

    def _depth_to_points(self, depth, R=None, t=None, fov=None, intrinsic=None) -> np.ndarray:
        """Adapted from https://github.com/isl-org/ZoeDepth/blob/edb6daf45458569e24f50250ef1ed08c015f17a7/zoedepth/utils/geometry.py#L39.
        """
        K = intrinsic
        Kinv = np.linalg.inv(K)
        if R is None:
            R = np.eye(3)
        if t is None:
            t = np.zeros(3)

        # M converts from your coordinate to PyTorch3D's coordinate system
        M = np.eye(3)

        height, width = depth.shape[1:3]

        x = np.arange(width)
        y = np.arange(height)
        coord = np.stack(np.meshgrid(x, y), -1)
        coord = np.concatenate((coord, np.ones_like(coord)[:, :, [0]]), -1)  # z=1
        coord = coord.astype(np.float32)
        coord = coord[None]  # bs, h, w, 3

        D = depth[:, :, :, None, None]
        pts3D_1 = D * Kinv[None, None, None, ...] @ coord[:, :, :, :, None]
        # pts3D_1 live in your coordinate system. Convert them to Py3D's
        pts3D_1 = M[None, None, None, ...] @ pts3D_1
        # from reference to targe tviewpoint
        pts3D_2 = R[None, None, None, ...] @ pts3D_1 + t[None, None, None, :, None]

        # G converts from your coordinate to PyTorch3D's coordinate system
        G = np.eye(3)
        G[0, 0] = -1.0
        G[1, 1] = -1.0

        return pts3D_2[:, :, :, :3, 0][0] @ G.T

    def _move_to_ground(self, pts3d: np.ndarray) -> np.ndarray:
        pts3d_flattened = pts3d.reshape(-1, 3)
        pts3d_sorted = pts3d_flattened[pts3d_flattened[:, 2].argsort()]
        fifty_percent_index = int(pts3d_sorted.shape[0] * 0.5)
        nearest_points = pts3d_sorted[:fifty_percent_index]
        min_y = np.array([0.0, np.min(nearest_points[:, 1]), 0.0])
        return pts3d - min_y, min_y

    def _create_object_pcd(self, pts3d: np.ndarray, image: Image.Image, mask: np.ndarray = None):
        if mask is not None:
            points = pts3d[mask].copy()
            colors = np.array(image)[mask] / 255.0
        else:
            points, colors = pts3d.copy().reshape((-1, 3)), np.array(image).reshape((-1, 3)) / 255.0

        # Perturb the points a bit to avoid colinearity
        points += np.random.normal(0, 4e-3, points.shape)

        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
        pcd.colors = o3d.utility.Vector3dVector(colors)
        return pcd

    def _process_pcd(self, pcd, run_dbscan=True):
        scale = np.linalg.norm(np.asarray(pcd.points).std(axis=0)) * 3.0 + 1e-6
        [pcd, _] = pcd.remove_statistical_outlier(nb_neighbors=10, std_ratio=1.2)
        pcd = pcd.voxel_down_sample(voxel_size=max(0.01, scale / 40))
        if run_dbscan:
            pcd = pcd_denoise_dbscan(
                pcd, eps=self.cfg.reconstruct3d.dbscan_eps,
                min_points=self.cfg.reconstruct3d.dbscan_min_points)
        return pcd

    def _get_axis_aligned_bounding_box(self, pcd):
        bbox = pcd.get_axis_aligned_bounding_box()
        min_coords, max_coords = bbox.get_min_bound(), bbox.get_max_bound()
        center = np.array([(min_val + max_val) / 2.0 for min_val, max_val in zip(min_coords, max_coords)])
        eulers = np.array([0.0, 0.0, 0.0])
        extent = np.array([max_val - min_val for min_val, max_val in zip(min_coords, max_coords)])
        return dict(
            center=center, eulers=eulers, extent=extent)

    def _get_oriented_bounding_box(self, pcd):
        bbox = pcd.get_oriented_bounding_box()
        center = np.asarray(bbox.center)
        eulers = Rotation.from_matrix(bbox.R.copy()).as_euler('XYZ')
        extent = np.asarray(bbox.extent)
        return dict(
            center=center, eulers=eulers, extent=extent)
