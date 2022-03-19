# -*- coding: utf-8 -*-
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

from detectron2.config import CfgNode as CN


def add_deepwindows_network_config(cfg):
    """
    Add new config for network.
    """
    cfg.MODEL.ROI_BOX_HEAD.GEOMETRIC_FEATURE = True
    cfg.MODEL.ROI_BOX_HEAD.NUM_RELATIONS = 32
    cfg.MODEL.ROI_BOX_HEAD.NUM_RM = [1,1]