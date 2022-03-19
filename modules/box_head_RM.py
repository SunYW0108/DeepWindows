# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import numpy as np
from typing import List, Tuple
# from typing import Dict, List, Optional, Tuple, Union
import fvcore.nn.weight_init as weight_init
import torch
from torch import nn
from torch.nn import functional as F

from detectron2.config import configurable
from detectron2.layers import Conv2d, Linear, ShapeSpec, get_norm
from detectron2.utils.registry import Registry

__all__ = ["FastRCNNConvFCHead_RM"]  # , "build_box_head", "ROI_BOX_HEAD_REGISTRY"]

from detectron2.modeling.roi_heads.box_head import ROI_BOX_HEAD_REGISTRY


def extract_position_matrix(pred_boxes: torch.Tensor, nongt_dim: int):
    """
    Extract position matrix
        Args:
            bbox: [num_boxes, 4]
        Returns:
            position_matrix: [num_boxes, nongt_dim, 4]
    """
    widths = pred_boxes[:, 2] - pred_boxes[:, 0]
    heights = pred_boxes[:, 3] - pred_boxes[:, 1]
    ctr_x = pred_boxes[:, 0] + 0.5 * widths
    ctr_y = pred_boxes[:, 1] + 0.5 * heights

    m_minimum = torch.full(ctr_x.size(), 1e-3).to(ctr_x.device)

    delta_x = torch.abs(ctr_x - torch.unsqueeze(ctr_x, 1))
    delta_x = delta_x / widths
    delta_x = torch.log(torch.where(delta_x > m_minimum, delta_x, m_minimum))

    delta_y = torch.abs(ctr_y - torch.unsqueeze(ctr_y, 1))
    delta_y = delta_y / heights
    delta_y = torch.log(torch.where(delta_y > m_minimum, delta_y, m_minimum))

    delta_w = torch.log(widths / torch.unsqueeze(widths, 1))
    delta_h = torch.log(heights / torch.unsqueeze(heights, 1))

    concat_list = [delta_x[:, :nongt_dim], delta_y[:, :nongt_dim], delta_w[:, :nongt_dim], delta_h[:, :nongt_dim]]
    position_matrix = torch.stack(concat_list, 2)
    return position_matrix


def extract_position_embedding(position_mat, feat_dim, wave_length=1000.0):
    """
    return a position_embedding [num_rois, nongt_dim,64]
    """
    feat_range = torch.arange(feat_dim / 8.0).to(position_mat.device)
    dim_mat = torch.pow(torch.full((1,), wave_length).to(position_mat.device), 8.0 / feat_dim * feat_range)
    dim_mat = dim_mat.view(1, 1, 1, -1)  # [1,1,1,8]
    position_mat = torch.unsqueeze(position_mat * 100.0, 3)
    div_mat = position_mat / dim_mat
    sin_mat = torch.sin(div_mat)
    cos_mat = torch.cos(div_mat)
    embedding = torch.cat((sin_mat, cos_mat), 3)
    embedding = embedding.view(embedding.size(0), embedding.size(1), -1)
    return embedding


def list_add(box_features, attention_features):
    add_feature = []
    for i in np.arange(len(box_features)):
        add_feature.append(box_features[i] + attention_features[i])
    return add_feature


@ROI_BOX_HEAD_REGISTRY.register()
class FastRCNNConvFCHead_RM(nn.Module):
    """
    A head with several 3x3 conv layers (each followed by norm & relu) and then
    several fc layers (each followed by relu).
    """

    @configurable
    def __init__(
            self, input_shape: ShapeSpec, *, conv_dims: List[int], fc_dims: List[int], conv_norm="",
            post_nms_topk: Tuple[int, int], use_geo_feat: bool, num_relations: int, num_rm: List[int],
            img_per_batch: int
    ):
        """
        NOTE: this interface is experimental.

        Args:
            input_shape (ShapeSpec): shape of the input feature.
            conv_dims (list[int]): the output dimensions of the conv layers
            fc_dims (list[int]): the output dimensions of the fc layers
            conv_norm (str or callable): normalization for the conv layers.
                See :func:`detectron2.layers.get_norm` for supported types.
        """
        super().__init__()
        assert len(conv_dims) + len(fc_dims) > 0

        self.USE_GEO_FEAT = use_geo_feat
        self.NUM_RELATIONS = num_relations
        self.POST_NMS_TOPK = {True: post_nms_topk[0], False: post_nms_topk[1]}
        self.NUM_RM = num_rm
        self.IMS_PER_BATCH = img_per_batch

        self._output_size = (input_shape.channels, input_shape.height, input_shape.width)

        self.conv_norm_relus = []
        for k, conv_dim in enumerate(conv_dims):
            conv = Conv2d(
                self._output_size[0],
                conv_dim,
                kernel_size=3,
                padding=1,
                bias=not conv_norm,
                norm=get_norm(conv_norm, conv_dim),
                activation=F.relu,
            )
            self.add_module("conv{}".format(k + 1), conv)
            self.conv_norm_relus.append(conv)
            self._output_size = (conv_dim, self._output_size[1], self._output_size[2])

        self.fcs = []
        for k, fc_dim in enumerate(fc_dims):
            fc = Linear(np.prod(self._output_size), fc_dim)
            self.add_module("fc{}".format(k + 1), fc)
            self.fcs.append(fc)
            self._output_size = fc_dim

        for layer in self.conv_norm_relus:
            weight_init.c2_msra_fill(layer)
        for layer in self.fcs:
            weight_init.c2_xavier_fill(layer)

        self.WG = []
        self.WK = []
        self.WQ = []
        self.WV = []
        index = 0
        for idx_fc in np.arange(len(fc_dims)):
            for idx_rm in np.arange(self.NUM_RM[idx_fc]):
                wg = Linear(64, self.NUM_RELATIONS)
                self.add_module("WG{}".format(index + 1), wg)
                self.WG.append(wg)

                wk = Linear(1024, 1024)
                self.add_module("WK{}".format(index + 1), wk)
                self.WK.append(wk)

                wq = Linear(1024, 1024)
                self.add_module("WQ{}".format(index + 1), wq)
                self.WQ.append(wq)

                wv = Conv2d(self.NUM_RELATIONS * 1024, 1024, kernel_size=1, groups=self.NUM_RELATIONS)
                self.add_module("WV{}".format(index + 1), wv)
                self.WV.append(wv)

                index = index + 1

        for layer in self.WG:
            nn.init.normal_(layer.weight, std=0.01)
            nn.init.constant_(layer.bias, 0)
        for layer in self.WK:
            nn.init.normal_(layer.weight, std=0.01)
            nn.init.constant_(layer.bias, 0)
        for layer in self.WQ:
            nn.init.normal_(layer.weight, std=0.01)
            nn.init.constant_(layer.bias, 0)
        for layer in self.WV:
            nn.init.normal_(layer.weight, std=0.01)
            nn.init.constant_(layer.bias, 0)

    @classmethod
    def from_config(cls, cfg, input_shape):
        num_conv = cfg.MODEL.ROI_BOX_HEAD.NUM_CONV
        conv_dim = cfg.MODEL.ROI_BOX_HEAD.CONV_DIM
        num_fc = cfg.MODEL.ROI_BOX_HEAD.NUM_FC
        fc_dim = cfg.MODEL.ROI_BOX_HEAD.FC_DIM

        post_nms_topk = (cfg.MODEL.RPN.POST_NMS_TOPK_TRAIN, cfg.MODEL.RPN.POST_NMS_TOPK_TEST)
        use_geo_feat = cfg.MODEL.ROI_BOX_HEAD.GEOMETRIC_FEATURE
        num_relations = cfg.MODEL.ROI_BOX_HEAD.NUM_RELATIONS
        num_rm = cfg.MODEL.ROI_BOX_HEAD.NUM_RM
        img_per_batch = cfg.SOLVER.IMS_PER_BATCH

        return {
            "input_shape": input_shape,
            "conv_dims": [conv_dim] * num_conv,
            "fc_dims": [fc_dim] * num_fc,
            "conv_norm": cfg.MODEL.ROI_BOX_HEAD.NORM,
            "post_nms_topk": post_nms_topk,
            "use_geo_feat": use_geo_feat,
            "num_relations": num_relations,
            "num_rm": num_rm,
            "img_per_batch": img_per_batch,
        }

    def attention_module_multi_head(
            self,
            attention_features: List[torch.tensor],
            position_embedding: List[torch.tensor],
            nongt_dims,
            fc_dim=16,
            feat_dim=1024,
            index=0,
            group=16,
            dim=(1024, 1024, 1024),
    ):
        """
        combine appearance feature and geometric feature
        Attetion module with vectorized version
        Args:
            roi_feat: [num_rois, feat_dim]
            position_embedding: [num_rois, nongt_dim, emb_dim]
            fc_dim: should be same as group
            feat_dim: dimension of roi_feat, should be same as dim[2]
            dim: a 3-tuple of (query, key, output)
            group:16
            index:
        Returns:
            output: [num_rois, ovr_feat_dim, output_dim]
        """
        attention_return = [
            self.attention_module_multi_head_per_img(attention_feature, embedding, nongt_dim, fc_dim, feat_dim, index,
                                                     group, dim)
            for attention_feature, embedding, nongt_dim in zip(attention_features, position_embedding, nongt_dims)
        ]
        return attention_return

    def attention_module_multi_head_per_img(
            self,
            attention_feature: torch.tensor,
            position_embedding: torch.tensor,
            nongt_dim,
            fc_dim=16,
            feat_dim=1024,
            index=0,
            group=16,
            dim=(1024, 1024, 1024),
    ):
        dim_group = (int(dim[0] / group), int(dim[1] / group), int(dim[2] / group))
        # print("dim_group:",dim_group)
        nongt_roi_feat = attention_feature[:nongt_dim, :]
        position_embedding = position_embedding.view(-1, position_embedding.size(2))
        position_feat_1_relu = F.relu(self.WG[index](position_embedding))
        aff_weight = position_feat_1_relu.view(-1, nongt_dim, fc_dim)
        aff_weight = aff_weight.permute(0, 2, 1)
        # print("aff_weight:",aff_weight.shape)

        q_data = self.WQ[index](attention_feature)
        q_data = q_data.view(-1, group, dim_group[0]).permute(1, 0, 2)
        # print("q_data:", q_data.shape)

        k_data = self.WK[index](nongt_roi_feat)
        k_data = k_data.view(-1, group, dim_group[1]).permute(1, 0, 2)
        # print("k_data:", k_data.shape)

        aff = torch.matmul(q_data, k_data.permute(0, 2, 1))
        aff_scale = (1.0 / np.sqrt(float(dim_group[1]))) * aff
        aff_scale = aff_scale.permute(1, 0, 2)
        # print("aff_scale:",aff_scale.shape)

        assert fc_dim == group, 'fc_dim !=group'

        aff_weight_minimum = torch.full(aff_weight.size(), 1e-6).to(aff_weight.device)
        weighted_aff = torch.log(torch.where(aff_weight > aff_weight_minimum, aff_weight, aff_weight_minimum))

        weighted_aff = weighted_aff + aff_scale
        # print("weighted_aff:",weighted_aff.shape)
        m_softmax = nn.Softmax(dim=2)
        aff_softmax = m_softmax(weighted_aff)
        aff_softmax_reshape = aff_softmax.view(-1, aff_softmax.size(2))
        output_t = torch.matmul(aff_softmax_reshape, nongt_roi_feat)
        output_t = output_t.view(-1, fc_dim * feat_dim, 1, 1)
        # print("output_t:",output_t.shape)
        linear_out = self.WV[index](output_t)
        # print("linear_out:",linear_out.shape)
        linear_out = linear_out.view(linear_out.size(0), linear_out.size(1))
        return linear_out

    def forward(self,
                box_features: torch.tensor,
                pred_boxes: List[torch.tensor],
                nongt_dims: List[int]):

        num_boxes = [boxes.size(0) for boxes in pred_boxes]
        box_features = list(torch.split(box_features, num_boxes, dim=0))

        if self.USE_GEO_FEAT:
            position_matrix = [extract_position_matrix(pred_boxes_per_img, nongt_dim_per_img)
                               for pred_boxes_per_img, nongt_dim_per_img in zip(pred_boxes, nongt_dims)]
            position_embedding = [extract_position_embedding(position_matrix_per_img, feat_dim=64)
                                  for position_matrix_per_img in position_matrix]
            # position_embedding = torch.cat(position_embedding)

        for layer in self.conv_norm_relus:
            box_features = [layer(feature) for feature in box_features]
        if len(self.fcs):
            if box_features[0].dim() > 2:
                box_features = [torch.flatten(feature, start_dim=1) for feature in box_features]
            attention_index = 0
            for idx_fc, layer in enumerate(self.fcs):
                # box_features = F.relu(layer(box_features))
                # important**********************************************************
                box_features = [layer(feature) for feature in box_features]
                if self.NUM_RM[idx_fc]:
                    attention_features = box_features
                    for idx_rm in np.arange(self.NUM_RM[idx_fc]):
                        attention_features = self.attention_module_multi_head(attention_features, position_embedding,
                                                                              nongt_dims,
                                                                              fc_dim=self.NUM_RELATIONS, feat_dim=1024,
                                                                              index=attention_index,
                                                                              group=self.NUM_RELATIONS,
                                                                              dim=(1024, 1024, 1024))
                        attention_index = attention_index + 1
                    box_features = list_add(box_features, attention_features)
                    box_features = [F.relu(feature) for feature in box_features]
                else:
                    box_features = [F.relu(feature) for feature in box_features]

        return torch.cat(box_features)

    @property
    def output_shape(self):
        """
        Returns:
            ShapeSpec: the output feature shape
        """
        o = self._output_size
        if isinstance(o, int):
            return ShapeSpec(channels=o)
        else:
            return ShapeSpec(channels=o[0], height=o[1], width=o[2])

# def build_box_head(cfg, input_shape):
#     """
#     Build a box head defined by `cfg.MODEL.ROI_BOX_HEAD.NAME`.
#     """
#     name = cfg.MODEL.ROI_BOX_HEAD.NAME
#     return ROI_BOX_HEAD_REGISTRY.get(name)(cfg, input_shape)
