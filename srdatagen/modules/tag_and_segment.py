from typing import Any, Dict, List

import cv2

import numpy as np
from PIL import Image
import supervision as sv
import torch
import torchvision

# GroundingDINO
from groundingdino.util.inference import Model as GroundingDINOModel

# RAM
from ram import get_transform, inference_ram
from ram.models import ram_plus

# SAM
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor

from srdatagen import AnnotType, SkipSampleException


class TagAndSegment:
    """Tag objects in an image and segment their masks.
    """

    def __init__(self, cfg, device):
        self.cfg = cfg
        self.device = device

        # Build RAM
        self.ram_model = ram_plus(
            pretrained=self.cfg.ram.pretrained_ckpt,
            image_size=384, vit='swin_l')
        self.ram_model.eval().to(self.device)
        self.ram_transform = get_transform(image_size=384)

        # Build GroundingDINO
        self.gdino_model = GroundingDINOModel(
            model_config_path=self.cfg.gdino.model_config_path,
            model_checkpoint_path=self.cfg.gdino.model_checkpoint_path,
            device=self.device)

        # Build SAM
        self.sam_model = build_sam2(self.cfg.sam.cfg_path, self.cfg.sam.ckpt_path).to(self.device)
        self.sam_predictor = SAM2ImagePredictor(self.sam_model)

    @torch.no_grad()
    def __call__(self, image: Image.Image, annot: AnnotType):
        # Step 1: Run RAM
        tags = self._run_ram(image)
        annot['tags'] = tags
        if len(tags) == 0:
            raise SkipSampleException('No tags detected by RAM')

        # Step 2: Run GroundingDINO
        detections = self._run_gdino(image, tags)
        annot['detections'] = detections
        if len(detections.class_id) < 1:
            raise SkipSampleException('No objects detected by GroundingDINO')

        # Step 3: Run SAM
        mask = self._run_sam(image, detections)
        detections.mask = mask

        # Step 4: Filtering
        detections = self._filter_detections(annot, detections)
        if len(detections.class_id) < 1:
            raise SkipSampleException('No objects after filtering')

        # Step 5: Sort detections by area
        sorted_indices = np.argsort(-detections.area)
        detections = detections[sorted_indices]

        # Step 6: Update masks to be subtracted from masks of contained bboxes
        mask_subtracted = self._mask_subtract_contained(detections)

        # Step 7: Prepare outputs
        detections_data = dict(
            xyxy=detections.xyxy, confidence=detections.confidence,
            class_id=detections.class_id, mask=detections.mask,
            box_area=detections.box_area, area=detections.area,
            mask_subtracted=mask_subtracted, classes=tags)
        annot['detections'] = self._prepare_outputs(detections_data)

        return annot

    def _run_ram(self, image: Image.Image) -> List[str]:
        image = image.copy()
        image = image.resize((384, 384))
        image = self.ram_transform(image).unsqueeze(0).to(self.device)

        res = inference_ram(image, self.ram_model)
        tags = [x.strip() for x in res[0].replace('|', ',').split(',')]
        tags = [x.lower() for x in tags if x != '']

        # Remove ignored classes
        tags = [x for x in tags if x not in self.cfg.ram.ignore_classes]

        return tags

    def _run_gdino(self, image: Image.Image, tags: List[str]) -> sv.Detections:
        gdino_input = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        detections = self.gdino_model.predict_with_classes(
            image=gdino_input, classes=tags,
            box_threshold=self.cfg.gdino.box_threshold,
            text_threshold=self.cfg.gdino.text_threshold)

        # Run NMS: https://github.com/IDEA-Research/Grounded-Segment-Anything
        nms_idx = torchvision.ops.nms(
            torch.from_numpy(detections.xyxy),
            torch.from_numpy(detections.confidence),
            self.cfg.gdino.nms_threshold).numpy().tolist()
        detections = detections[nms_idx]

        detections = detections[detections.class_id != -1]

        return detections

    def _run_sam(self, image: Image.Image, detections: sv.Detections) -> np.ndarray:
        sam_input = np.array(image)
        self.sam_predictor.set_image(sam_input)
        all_masks = []
        for box in detections.xyxy:
            # Choose the mask with the highest score out of three masks
            masks, scores, _ = self.sam_predictor.predict(box=box, multimask_output=True)
            masks = masks > 0.0
            all_masks.append(masks[np.argmax(scores)])
        return np.array(all_masks)

    def _filter_detections(self, annot: AnnotType, detections: sv.Detections) -> sv.Detections:
        h, w = annot['image_info']['height_resized'], annot['image_info']['width_resized']
        valid_idx = []

        for obj_idx in range(len(detections.class_id)):
            area_ratio = detections.mask[obj_idx].sum() / h / w
            if area_ratio < self.cfg.filter.min_mask_area_ratio:
                continue
            if area_ratio > self.cfg.filter.max_mask_area_ratio:
                continue
            if detections.confidence[obj_idx] < self.cfg.filter.mask_confidence_threshold:
                continue
            valid_idx.append(obj_idx)

        return detections[valid_idx]

    def _mask_subtract_contained(self, detections: sv.Detections, th1=0.8, th2=0.7) -> sv.Detections:
        """Adapted from: https://github.com/concept-graphs/concept-graphs/blob/93277a02bd89171f8121e84203121cf7af9ebb5d/conceptgraph/utils/ious.py#L453
        """
        xyxy = detections.xyxy
        mask = detections.mask
        areas = (xyxy[:, 2] - xyxy[:, 0]) * (xyxy[:, 3] - xyxy[:, 1])

        # Compute intersection boxes
        lt = np.maximum(xyxy[:, None, :2], xyxy[None, :, :2])  # left-top points (N, N, 2)
        rb = np.minimum(xyxy[:, None, 2:], xyxy[None, :, 2:])  # right-bottom points (N, N, 2)

        inter = (rb - lt).clip(min=0)  # intersection sizes (dx, dy), if no overlap, clamp to zero (N, N, 2)

        # Compute areas of intersection boxes
        inter_areas = inter[:, :, 0] * inter[:, :, 1]  # (N, N)

        inter_over_box1 = inter_areas / areas[:, None]  # (N, N)
        # inter_over_box2 = inter_areas / areas[None, :] # (N, N)
        inter_over_box2 = inter_over_box1.T  # (N, N)

        # if the intersection area is smaller than th2 of the area of box1,
        # and the intersection area is larger than th1 of the area of box2,
        # then box2 is considered contained by box1
        contained = (inter_over_box1 < th2) & (inter_over_box2 > th1)  # (N, N)
        contained_idx = contained.nonzero()  # (num_contained, 2)

        mask_sub = mask.copy()  # (N, H, W)
        # mask_sub[contained_idx[0]] = mask_sub[contained_idx[0]] & (~mask_sub[contained_idx[1]])
        for i in range(len(contained_idx[0])):
            mask_sub[contained_idx[0][i]] = mask_sub[contained_idx[0][i]] & (~mask_sub[contained_idx[1][i]])

        return mask_sub

    def _prepare_outputs(self, detections_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        output = []
        for i in range(len(detections_data['xyxy'])):
            output.append(dict(
                object_name=f'obj_{i:02d}_{detections_data["classes"][detections_data["class_id"][i]]}',
                class_name=detections_data['classes'][detections_data['class_id'][i]],
                xyxy=detections_data['xyxy'][i],
                confidence=detections_data['confidence'][i].item(),
                class_id=detections_data['class_id'][i].item(),
                box_area=detections_data['box_area'][i].item(),
                area=detections_data['area'][i].item(),
                mask=detections_data['mask'][i],
                mask_subtracted=detections_data['mask_subtracted'][i]))
        return output
