# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
from .config import add_deepwindows_network_config
from .box_head_RM import FastRCNNConvFCHead_RM
from .roi_heads_RM import StandardROIHeads_RM
from .rpn_casa import CASA_RPNHead