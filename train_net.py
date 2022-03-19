#!/usr/bin/env python
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
Detection Training Script.

This scripts reads a given config file and runs the training or evaluation.
It is an entry point that is made to train standard models in detectron2.

In order to let one script support training of many models,
this script contains logic that are specific to these built-in models and therefore
may not be suitable for your own project.
For example, your research project perhaps only needs a single "evaluator".

Therefore, we recommend you to use detectron2 as an library and take
this file as an example of how to use the library.
You may want to write your own script with your datasets and other customizations.
"""

import logging
import os
from collections import OrderedDict
import torch
import time
import detectron2.utils.comm as comm
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.config import get_cfg
from detectron2.data import MetadataCatalog
from detectron2.engine import DefaultTrainer, default_argument_parser, default_setup, hooks, launch
from detectron2.evaluation import (
    CityscapesInstanceEvaluator,
    CityscapesSemSegEvaluator,
    COCOEvaluator,
    COCOPanopticEvaluator,
    DatasetEvaluators,
    LVISEvaluator,
    PascalVOCDetectionEvaluator,
    SemSegEvaluator,
    verify_results,
)
from detectron2.modeling import GeneralizedRCNNWithTTA
from detectron2.data.datasets import register_coco_instances
from modules import add_deepwindows_network_config


class Trainer(DefaultTrainer):
    """
    We use the "DefaultTrainer" which contains pre-defined default logic for
    standard training workflow. They may not work for you, especially if you
    are working on a new research project. In that case you can write your
    own training loop. You can use "tools/plain_train_net.py" as an example.
    """

    @classmethod
    def build_evaluator(cls, cfg, dataset_name, output_folder=None):
        """
        Create evaluator(s) for a given dataset.
        This uses the special metadata "evaluator_type" associated with each builtin dataset.
        For your own dataset, you can simply create an evaluator manually in your
        script and do not have to worry about the hacky if-else logic here.
        """
        if output_folder is None:
            output_folder = os.path.join(cfg.OUTPUT_DIR, "inference")
        return COCOEvaluator(dataset_name, cfg, True, output_folder)

    @classmethod
    def test_with_TTA(cls, cfg, model):
        logger = logging.getLogger("detectron2.trainer")
        # In the end of training, run an evaluation with TTA
        # Only support some R-CNN models.
        logger.info("Running inference with test-time augmentation ...")
        model = GeneralizedRCNNWithTTA(cfg, model)
        evaluators = [
            cls.build_evaluator(
                cfg, name, output_folder=os.path.join(cfg.OUTPUT_DIR, "inference_TTA")
            )
            for name in cfg.DATASETS.TEST
        ]
        res = cls.test(cfg, model, evaluators)
        res = OrderedDict({k + "_TTA": v for k, v in res.items()})
        return res


def setup(args):
    """
    Create configs and perform basic setups.
    """
    cfg = get_cfg()
    add_deepwindows_network_config(cfg)
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)

    # cfg.MODEL.WEIGHTS = ("./output/model_final.pth")

    cfg.freeze()
    default_setup(cfg, args)
    return cfg

def register_dataset():
    metadata = {'thing_classes': ['window'], }

    # dataset path
    TRAIN_PATH = '/home/sun/facades_datasets/ALL/train'
    VAL_PATH = '/home/sun/facades_datasets/ALL/val'

    register_coco_instances("CMP_TRAIN", metadata,
                            os.path.join(TRAIN_PATH, 'Window_CMP_XML_TRAIN.json'),
                            os.path.join(TRAIN_PATH, 'images'))
    register_coco_instances("CMP_VAL", metadata,
                            os.path.join(VAL_PATH, 'Window_CMP_XML_VAL.json'),
                            os.path.join(VAL_PATH, 'images'))

    register_coco_instances("CMP_BASE_TRAIN", metadata,
                            os.path.join(TRAIN_PATH, 'Window_CMPB_XML_TRAIN.json'),
                            os.path.join(TRAIN_PATH, 'images'))
    register_coco_instances("CMP_BASE_VAL", metadata,
                            os.path.join(VAL_PATH, 'Window_CMPB_XML_VAL.json'),
                            os.path.join(VAL_PATH, 'images'))

    register_coco_instances("CMP_extended_TRAIN", metadata,
                            os.path.join(TRAIN_PATH, 'Window_CMPE_XML_TRAIN.json'),
                            os.path.join(TRAIN_PATH, 'images'))
    register_coco_instances("CMP_extended_VAL", metadata,
                            os.path.join(VAL_PATH, 'Window_CMPE_XML_VAL.json'),
                            os.path.join(VAL_PATH, 'images'))

    register_coco_instances("graz50_TRAIN", metadata,
                            os.path.join(TRAIN_PATH, 'Window_GRAZ50_TRAIN.json'),
                            os.path.join(TRAIN_PATH, 'images'))
    register_coco_instances("graz50_VAL", metadata,
                            os.path.join(VAL_PATH, 'Window_GRAZ50_VAL.json'),
                            os.path.join(VAL_PATH, 'images'))

    register_coco_instances("etrims_TRAIN", metadata,
                            os.path.join(TRAIN_PATH, 'Window_etrims_TRAIN.json'),
                            os.path.join(TRAIN_PATH, 'images'))
    register_coco_instances("etrims_VAL", metadata,
                            os.path.join(VAL_PATH, 'Window_etrims_VAL.json'),
                            os.path.join(VAL_PATH, 'images'))

    register_coco_instances("varcity3d_TRAIN", metadata,
                            os.path.join(TRAIN_PATH, 'Window_varcity3d_TRAIN.json'),
                            os.path.join(TRAIN_PATH, 'images'))
    register_coco_instances("varcity3d_VAL", metadata,
                            os.path.join(VAL_PATH, 'Window_varcity3d_VAL.json'),
                            os.path.join(VAL_PATH, 'images'))

    register_coco_instances("ECP_TRAIN", metadata,
                            os.path.join(TRAIN_PATH, 'Window_ECP_TRAIN.json'),
                            os.path.join(TRAIN_PATH, 'images'))
    register_coco_instances("ECP_VAL", metadata,
                            os.path.join(VAL_PATH, 'Window_ECP_VAL.json'),
                            os.path.join(VAL_PATH, 'images'))

    register_coco_instances("ParisArtDeco_TRAIN", metadata,
                            os.path.join(TRAIN_PATH, 'Window_ParisArtDeco_TRAIN.json'),
                            os.path.join(TRAIN_PATH, 'images'))
    register_coco_instances("ParisArtDeco_VAL", metadata,
                            os.path.join(VAL_PATH, 'Window_ParisArtDeco_VAL.json'),
                            os.path.join(VAL_PATH, 'images'))

    register_coco_instances("TUBS_TRAIN", metadata,
                            os.path.join(TRAIN_PATH, 'Window_TUBS_TRAIN.json'),
                            os.path.join(TRAIN_PATH, 'images'))
    register_coco_instances("TUBS_VAL", metadata,
                            os.path.join(VAL_PATH, 'Window_TUBS_VAL.json'),
                            os.path.join(VAL_PATH, 'images'))

    register_coco_instances("All_TRAIN", metadata,
                            os.path.join(TRAIN_PATH, 'Window_Instances_ALL_TRAIN.json'),
                            os.path.join(TRAIN_PATH, 'images'))
    register_coco_instances("All_VAL", metadata,
                            os.path.join(VAL_PATH, 'Window_Instances_ALL_VAL.json'),
                            os.path.join(VAL_PATH, 'images'))


def main(args):
    cfg = setup(args)

    register_dataset()

    if args.eval_only:
        model = Trainer.build_model(cfg)
        DetectionCheckpointer(model, save_dir=cfg.OUTPUT_DIR).resume_or_load(
            cfg.MODEL.WEIGHTS, resume=args.resume
        )
        res = Trainer.test(cfg, model)
        if cfg.TEST.AUG.ENABLED:
            res.update(Trainer.test_with_TTA(cfg, model))
        if comm.is_main_process():
            verify_results(cfg, res)
        return res

    """
    If you'd like to do anything fancier than the standard training logic,
    consider writing your own training loop (see plain_train_net.py) or
    subclassing the trainer.
    """
    trainer = Trainer(cfg)
    trainer.resume_or_load(resume=args.resume)
    if cfg.TEST.AUG.ENABLED:
        trainer.register_hooks(
            [hooks.EvalHook(0, lambda: trainer.test_with_TTA(cfg, trainer.model))]
        )
    return trainer.train()


if __name__ == "__main__":
    args = default_argument_parser().parse_args()
    print("Command Line Args:", args)
    launch(
        main,
        args.num_gpus,
        num_machines=args.num_machines,
        machine_rank=args.machine_rank,
        dist_url=args.dist_url,
        args=(args,),
    )
