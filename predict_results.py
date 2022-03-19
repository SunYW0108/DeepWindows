# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import argparse
import torch
import glob
import multiprocessing as mp
import os
import time
import cv2
import tqdm
import numpy as np
from detectron2.utils.visualizer import GenericMask, ColorMode
from detectron2.structures import BitMasks, Boxes, BoxMode, Keypoints, PolygonMasks, RotatedBoxes

from detectron2.config import get_cfg
from detectron2.data.detection_utils import read_image
from detectron2.utils.logger import setup_logger

from predictor import VisualizationDemo
from train_net import register_dataset
from modules import add_deepwindows_network_config
from detectron2.utils.visualizer import Visualizer

# constants
WINDOW_NAME = "COCO detections"

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
            masks = np.asarray(predictions.pred_masks.cpu())
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


def setup_cfg(args):
    # load config from file and command-line arguments
    cfg = get_cfg()
    add_deepwindows_network_config(cfg)
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)

    # Set score_threshold for builtin models
    cfg.MODEL.RETINANET.SCORE_THRESH_TEST = args.confidence_threshold
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = args.confidence_threshold
    cfg.MODEL.PANOPTIC_FPN.COMBINE.INSTANCES_CONFIDENCE_THRESH = args.confidence_threshold
    cfg.freeze()
    return cfg


def get_parser():
    parser = argparse.ArgumentParser(description="Detectron2 demo for builtin models")
    parser.add_argument(
        "--config-file",
        default="configs/quick_schedules/mask_rcnn_R_50_FPN_inference_acc_test.yaml",
        metavar="FILE",
        help="path to config file",
    )
    parser.add_argument("--webcam", action="store_true", help="Take inputs from webcam.")
    parser.add_argument("--video-input", help="Path to video file.")
    parser.add_argument(
        "--input",
        # nargs="+",
        help="A list of space separated input images; "
        "or a single glob pattern such as 'directory/*.jpg'",
    )
    parser.add_argument(
        "--output",
        help="A file or directory to save output visualizations. "
        "If not given, will show output in an OpenCV window.",
    )

    parser.add_argument(
        "--confidence-threshold",
        type=float,
        default=0.5,
        help="Minimum score for instance predictions to be shown",
    )
    parser.add_argument(
        "--opts",
        help="Modify config options using the command-line 'KEY VALUE' pairs",
        default=[],
        nargs=argparse.REMAINDER,
    )
    return parser


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    args = get_parser().parse_args()
    setup_logger(name="fvcore")
    logger = setup_logger()
    logger.info("Arguments: " + str(args))

    cfg = setup_cfg(args)

    register_dataset()

    demo = VisualizationDemo(cfg)

    if args.input:
        # if len(args.input) == 1:
        #     args.input = glob.glob(os.path.expanduser(args.input[0]))
        #     assert args.input, "The input path(s) was not found"
        # for path in tqdm.tqdm(args.input, disable=not args.output):
        for imgfile in os.listdir(args.input):
            # if os.path.splitext(imgfile)[-1] != ".jpg":
            #     continue
            # use PIL, to be consistent with evaluation
            img_fullName = os.path.join(args.input, imgfile)
            img = read_image(img_fullName, format="BGR")
            start_time = time.time()
            predictions, visualized_output = demo.run_on_image(img)
            logger.info(
                "{}: {} in {:.2f}s".format(
                    imgfile,
                    "detected {} instances".format(len(predictions["instances"]))
                    if "instances" in predictions
                    else "finished",
                    time.time() - start_time,
                )
            )

            if args.output:
                if os.path.isdir(args.output):
                    assert os.path.isdir(args.output), args.output
                    out_filename = os.path.join(args.output, os.path.basename(imgfile))
                else:
                    assert len(args.input) == 1, "Please specify a directory with args.output"
                    out_filename = args.output
                visualized_output.save(out_filename)
            else:
                cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
                cv2.imshow(WINDOW_NAME, visualized_output.get_image()[:, :, ::-1])
                if cv2.waitKey(0) == 27:
                    break  # esc to quit