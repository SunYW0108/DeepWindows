#!/usr/bin/env python
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

import argparse
import json
import numpy as np
import os
from collections import defaultdict
import cv2
import tqdm
from fvcore.common.file_io import PathManager
import torch

from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.structures import Boxes, BoxMode, Instances
from detectron2.utils.logger import setup_logger
from detectron2.utils.visualizer import Visualizer
from detectron2.utils.visualizer import GenericMask, ColorMode
from train_net import register_dataset
from detectron2.structures import BitMasks, Boxes, BoxMode, Keypoints, PolygonMasks, RotatedBoxes


class drawBoolMap(Visualizer):

    def __init__(self, cfg, input_shape):
        super(drawBoolMap, self).__init__(cfg, input_shape)

        self.output = np.zeros(self.img.shape[:2], dtype=np.uint8)

    def _convert_masks(self, masks_or_polygons):
        """
        Convert different format of masks or polygons to a tuple of masks and polygons.

        Returns:
            list[GenericMask]:
        """
        m = masks_or_polygons
        if isinstance(m, PolygonMasks):
            m = m.polygons
        if isinstance(m, BitMasks):
            m = m.tensor.numpy()
        if isinstance(m, torch.Tensor):
            m = m.numpy()
        ret = []
        for x in m:
            if isinstance(x, GenericMask):
                ret.append(x)
            else:
                ret.append(GenericMask(x, self.output.shape[0], self.output.shape[1]))
        return ret

    def draw_instance_predictions(self, predictions):
        """
        Draw instance-level prediction results on an image.
        """
        num_instances = None
        if predictions.has("pred_masks"):
            masks = np.asarray(predictions.pred_masks)
            masks = [GenericMask(x, self.output.shape[0], self.output.shape[1]) for x in masks]
        else:
            masks = None

        if masks is not None:
            masks = self._convert_masks(masks)
            if num_instances:
                assert len(masks) == num_instances
            else:
                num_instances = len(masks)

        for i in range(num_instances):
            if masks is not None:
                for segment in masks[i].polygons:
                    cv2.fillPoly(self.output, [np.round(segment.reshape(-1, 2)).astype(int)], 1)
        return self.output

    def draw_dataset_dict(self, dic):
        """
        Draw annotations/segmentaions in Detectron2 Dataset format.
        """
        num_instances = None
        annos = dic.get("annotations", None)
        if annos:
            if "segmentation" in annos[0]:
                masks = [x["segmentation"] for x in annos]
            else:
                masks = None

            if masks is not None:
                masks = self._convert_masks(masks)
                if num_instances:
                    assert len(masks) == num_instances
                else:
                    num_instances = len(masks)

            for i in range(num_instances):
                if masks is not None:
                    for segment in masks[i].polygons:
                        cv2.fillPoly(self.output, [np.round(segment.reshape(-1, 2)).astype(int)], 1)
        return self.output


def create_instances(predictions, image_size):
    ret = Instances(image_size)

    score = np.asarray([x["score"] for x in predictions])
    chosen = (score > args.conf_threshold).nonzero()[0]
    score = score[chosen]  # choose indices (score>0.5)
    bbox = np.asarray([predictions[i]["bbox"] for i in chosen]).reshape(-1, 4)
    bbox = BoxMode.convert(bbox, BoxMode.XYWH_ABS, BoxMode.XYXY_ABS)

    labels = np.asarray([dataset_id_map(predictions[i]["category_id"]) for i in chosen])

    ret.scores = score
    ret.pred_boxes = Boxes(bbox)
    ret.pred_classes = labels

    try:
        ret.pred_masks = [predictions[i]["segmentation"] for i in chosen]
    except KeyError:
        pass
    return ret


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="A script that visualizes the json predictions from COCO or LVIS dataset."
    )
    parser.add_argument("--input", required=True, help="JSON file produced by the model")
    parser.add_argument("--dataset", help="name of the dataset", default="coco_2017_val")
    parser.add_argument("--conf-threshold", default=0.5, type=float, help="confidence threshold")
    args = parser.parse_args()

    register_dataset()

    logger = setup_logger()

    with PathManager.open(args.input, "r") as f:
        predictions = json.load(f)

    pred_by_image = defaultdict(list)
    for p in predictions:
        pred_by_image[p["image_id"]].append(p)

    dicts = list(DatasetCatalog.get(args.dataset))  # GT
    metadata = MetadataCatalog.get(args.dataset)
    if hasattr(metadata, "thing_dataset_id_to_contiguous_id"):

        def dataset_id_map(ds_id):
            return metadata.thing_dataset_id_to_contiguous_id[ds_id]

    elif "lvis" in args.dataset:
        # LVIS results are in the same format as COCO results, but have a different
        # mapping from dataset category id to contiguous category id in [0, #categories - 1]
        def dataset_id_map(ds_id):
            return ds_id - 1

    else:
        raise ValueError("Unsupported dataset: {}".format(args.dataset))

    TPTN_PixelCount = 0
    TOTAL_PixelCount = 0
    for dic in tqdm.tqdm(dicts):
        img = cv2.imread(dic["file_name"], cv2.IMREAD_COLOR)[:, :, ::-1]
        basename = os.path.basename(dic["file_name"])
        TOTAL_PixelCount=TOTAL_PixelCount+img.shape[0]*img.shape[1]

        predictions = create_instances(pred_by_image[dic["image_id"]], img.shape[:2])
        # draw pred_wins
        vis = drawBoolMap(img, metadata)
        vis_pred = vis.draw_instance_predictions(predictions)
        # draw GT_wins
        vis = drawBoolMap(img, metadata)
        vis_gt = vis.draw_dataset_dict(dic)

        predBoolMap = (vis_pred == 1)
        gtBoolMap = (vis_gt == 1)
        interArea = np.logical_and(predBoolMap, gtBoolMap)
        unionArea=np.logical_or(predBoolMap, gtBoolMap)

        TPTN_PixelCount = TPTN_PixelCount + interArea.sum()+img.shape[0]*img.shape[1]-unionArea.sum()

    print(TPTN_PixelCount)
    print(TOTAL_PixelCount)
    print(1.0 * TPTN_PixelCount / TOTAL_PixelCount)
