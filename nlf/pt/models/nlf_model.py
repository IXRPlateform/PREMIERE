from typing import Dict, Optional

import numpy as np
import simplepyutils as spu
import torch
import torch.nn as nn
import torch.nn.functional as F

from nlf.paths import PROJDIR
from nlf.pt import ptu, ptu3d
from nlf.pt.models import util as model_util
from nlf.pt.util import get_config


class NLFModel(nn.Module):
    def __init__(self, backbone, weight_field):
        super().__init__()
        FLAGS = get_config()

        self.backbone = backbone
        self.heatmap_head = LocalizerHead(weight_field)
        self.input_resolution = FLAGS.proc_side

        joint_info = spu.load_pickle(f'{PROJDIR}/joint_info_866.pkl')
        i_left_joints = [i for i, n in enumerate(joint_info.names) if n[0] == 'l']
        i_right_joints = [joint_info.ids['r' + joint_info.names[i][1:]] for i in i_left_joints]
        i_center_joints = [i for i in range(joint_info.n_joints) if
                           i not in i_left_joints and
                           i not in i_right_joints]
        permutation = torch.tensor(
            i_left_joints + i_right_joints + i_center_joints, dtype=torch.int32)
        self.inv_permutation = torch.argsort(permutation)

        self.canonical_lefts = nn.Parameter(
            torch.zeros((len(i_left_joints), 3), dtype=torch.float32))
        self.canonical_centers = nn.Parameter(
            torch.zeros((len(i_center_joints), 2), dtype=torch.float32))

        self.canonical_locs_init = torch.tensor(
            np.load(f'{PROJDIR}/canonical_loc_symmetric_init_866.npy'), dtype=torch.float32)
        self.canonical_delta_mask = torch.tensor(
            [not is_hand_joint(n) for n in joint_info.names], dtype=torch.float32)

    @torch.jit.export
    def canonical_locs(self):
        canonical_rights = torch.cat([
            -1 * self.canonical_lefts[:, :1],
            self.canonical_lefts[:, 1:]], dim=1)
        canonical_centers = torch.cat([
            torch.zeros_like(self.canonical_centers[:, :1]),
            self.canonical_centers], dim=1)
        permuted = torch.cat([self.canonical_lefts, canonical_rights, canonical_centers],
                             dim=0)
        return (permuted.index_select(0, self.inv_permutation) *
                self.canonical_delta_mask[:, None] +
                self.canonical_locs_init)

    @torch.jit.export
    def predict_multi_same_canonicals(
            self, image: torch.Tensor, intrinsic_matrix: torch.Tensor,
            canonical_points: torch.Tensor):  # , flip_canonicals_per_image=()):
        features = self.backbone(image)
        coords2d, coords3d, uncertainties = self.heatmap_head.predict_same_canonicals(
            features, canonical_points)
        return self.heatmap_head.reconstruct_absolute(
            coords2d, coords3d, uncertainties, intrinsic_matrix)

    @torch.jit.export
    def get_features(self, image: torch.Tensor):
        f = self.backbone(image)
        return self.heatmap_head.layer(f)

    @torch.jit.export
    def predict_multi_same_weights(
            self, image: torch.Tensor, intrinsic_matrix: torch.Tensor,
            weights: Dict[str, torch.Tensor], flip_canonicals_per_image: torch.Tensor):
        features_processed = self.get_features(image)
        coords2d, coords3d, uncertainties = self.heatmap_head.decode_features_multi_same_weights(
            features_processed, weights, flip_canonicals_per_image)
        return self.heatmap_head.reconstruct_absolute(
            coords2d, coords3d, uncertainties, intrinsic_matrix)

    @torch.jit.export
    def get_weights_for_canonical_points(self, canonical_points: torch.Tensor):
        return self.heatmap_head.get_weights_for_canonical_points(canonical_points)


class LocalizerHead(nn.Module):
    def __init__(self, weight_field):
        super().__init__()
        FLAGS = get_config()
        self.uncert_bias = FLAGS.uncert_bias
        self.uncert_bias2 = FLAGS.uncert_bias2
        self.depth = FLAGS.depth
        self.weight_field = weight_field
        self.stride_test = FLAGS.stride_test
        self.centered_stride = FLAGS.centered_stride
        self.box_size_m = FLAGS.box_size_m
        self.proc_side = FLAGS.proc_side
        self.backbone_link_dim = FLAGS.backbone_link_dim
        self.fix_uncert_factor = FLAGS.fix_uncert_factor
        self.mix_3d_inside_fov = FLAGS.mix_3d_inside_fov
        self.weak_perspective = FLAGS.weak_perspective
        conv = nn.LazyConv2d(out_channels=self.backbone_link_dim, kernel_size=1, bias=False)
        self.layer = nn.Sequential(
            conv,
            nn.BatchNorm2d(self.backbone_link_dim, momentum=0.1, eps=1e-3),
            nn.SiLU())

    @torch.jit.export
    def forward(self, features: torch.Tensor, canonical_positions: Optional[torch.Tensor] = None):
        assert canonical_positions is not None
        weights = self.weight_field(canonical_positions)  # NP[C(c+1)]
        return self.call_with_weights(features, weights)

    @torch.jit.export
    def call_with_weights(self, features, weights):
        features_processed = self.layer(features)  # NHWc
        coords2d, coords3d_rel_pred, uncertainties = self.apply_weights3d(
            features_processed, weights, n_out_channels=2 + self.depth)
        coords2d_pred = model_util.heatmap_to_image(
            coords2d, self.proc_side, self.stride_test, self.centered_stride)
        coords3d_rel_pred = model_util.heatmap_to_metric(
            coords3d_rel_pred, self.proc_side, self.stride_test, self.centered_stride,
            self.box_size_m)
        return coords2d_pred, coords3d_rel_pred, uncertainties

    @torch.jit.export
    def predict_same_canonicals(self, features: torch.Tensor, canonical_positions: torch.Tensor):
        weights = self.weight_field(canonical_positions)  # NP[C(c+1)]
        features_processed = self.layer(features)  # NcHW

        coords2d, coords3d_rel_pred, uncertainties = self.apply_weights3d_same_canonicals(
            features_processed, weights)
        coords2d_pred = model_util.heatmap_to_image(
            coords2d, self.proc_side, self.stride_test, self.centered_stride)
        coords3d_rel_pred = model_util.heatmap_to_metric(
            coords3d_rel_pred, self.proc_side, self.stride_test, self.centered_stride,
            self.box_size_m)
        return coords2d_pred, coords3d_rel_pred, uncertainties

    @torch.jit.export
    def apply_weights3d(self, features: torch.Tensor, weights: torch.Tensor, n_out_channels: int):
        # features: NcHW 128,1280,8,8
        # weights:  NP[(c+1)C] 128,768,10*1281
        weights = weights.to(features.dtype)
        weights_resh = torch.unflatten(
            weights, -1, (features.shape[1] + 1, n_out_channels))  # NPC(c+1)
        w_tensor = weights_resh[..., :-1, :]  # NPCc
        b_tensor = weights_resh[..., -1, :]  # NPC
        # TODO: check if a different einsum order would be faster
        logits = (torch.einsum(
            'nchw,npcC->npChw', features, w_tensor) +
                  b_tensor[:, :, :, None, None])

        uncertainty_map = logits[:, :, 0].float()
        coords_metric_xy = ptu.soft_argmax(logits[:, :, 1].float(), dim=[3, 2])
        heatmap25d = ptu.softmax(logits[:, :, 2:].float(), dim=[4, 3, 2])
        heatmap2d = torch.sum(heatmap25d, dim=4)
        uncertainties = torch.sum(uncertainty_map * heatmap2d.detach(), dim=[3, 2])
        uncertainties = F.softplus(uncertainties + self.uncert_bias) + self.uncert_bias2
        coords25d = ptu.decode_heatmap(heatmap25d, dim=[4, 3, 2])
        coords2d = coords25d[..., :2]
        coords3d = torch.cat([coords_metric_xy, coords25d[..., 2:]], dim=-1)
        return coords2d, coords3d, uncertainties

    @torch.jit.export
    def transpose_weights(
            self, weights: torch.Tensor, n_in_channels: int, feature_dtype: torch.dtype):
        n_out_channels = 2 + self.depth
        weights = weights.to(feature_dtype)
        weights_resh = torch.unflatten(weights, -1, (n_in_channels + 1, n_out_channels))  # P(c+1)C
        w_tensor = weights_resh[..., :-1, :]  # PcC
        b_tensor = weights_resh[..., -1, :]  # PC
        w_tensor = w_tensor.permute(1, 0, 2)  # PcC-> cPC
        return w_tensor, b_tensor

    @torch.jit.export
    def apply_weights3d_same_canonicals(
            self, features: torch.Tensor, weights: torch.Tensor):
        # features: NcHW 128,1280,8,8
        # weights:  P[C(c+1)] 768,10*1281
        w_tensor, b_tensor = self.transpose_weights(weights, features.shape[1], features.dtype)
        return self.apply_weights3d_same_canonicals_impl(features, w_tensor, b_tensor)

    @torch.jit.export
    def apply_weights3d_same_canonicals_impl(
            self, features: torch.Tensor, w_tensor: torch.Tensor, b_tensor: torch.Tensor):
        n_out_channels = 2 + self.depth
        w_tensor = torch.flatten(w_tensor, start_dim=-2, end_dim=-1).mT.unsqueeze(-1).unsqueeze(-1)
        w_tensor = w_tensor.contiguous()
        b_tensor = b_tensor.reshape(-1)

        logits = F.conv2d(features, w_tensor, bias=b_tensor)
        logits = torch.unflatten(logits, 1, (-1, n_out_channels))  # npChw
        uncertainty_map = logits[:, :, 0].float()

        coords_metric_xy = ptu.soft_argmax(logits[:, :, 1].float(), dim=[3, 2])
        heatmap25d = ptu.softmax(logits[:, :, 2:].float(), dim=[4, 3, 2])
        heatmap2d = torch.sum(heatmap25d, dim=2)
        uncertainties = torch.sum(uncertainty_map * heatmap2d.detach(), dim=[3, 2])
        uncertainties = F.softplus(uncertainties + self.uncert_bias) + self.uncert_bias2
        coords25d = ptu.decode_heatmap(heatmap25d, dim=[4, 3, 2])
        coords2d = coords25d[..., :2]
        coords3d = torch.cat([coords_metric_xy, coords25d[..., 2:]], dim=-1)
        return coords2d, coords3d, uncertainties

    @torch.jit.export
    def get_weights_for_canonical_points(self, canonical_points: torch.Tensor):
        weights = self.weight_field(canonical_points)
        w_tensor, b_tensor = self.transpose_weights(
            weights, self.backbone_link_dim, torch.float16)
        weights_fl = self.weight_field(
            canonical_points * torch.tensor([-1, 1, 1], dtype=torch.float32,
                                            device=canonical_points.device))
        w_tensor_fl, b_tensor_fl = self.transpose_weights(
            weights_fl, self.backbone_link_dim, torch.float16)
        return dict(
            w_tensor=w_tensor, b_tensor=b_tensor,
            w_tensor_flipped=w_tensor_fl, b_tensor_flipped=b_tensor_fl)

    @torch.jit.export
    def decode_features_multi_same_weights(
            self, features: torch.Tensor, weights: Dict[str, torch.Tensor],
            flip_canonicals_per_image: torch.Tensor):
        features_processed = features
        flip_canonicals_per_image_ind = flip_canonicals_per_image.to(torch.int32)

        nfl_features_processed, fl_features_processed = ptu.dynamic_partition(
            features_processed, flip_canonicals_per_image_ind, 2)
        partitioned_indices = ptu.dynamic_partition(
            torch.arange(features_processed.shape[0], device=flip_canonicals_per_image_ind.device),
            flip_canonicals_per_image_ind, 2)
        nfl_coords2d, nfl_coords3d, nfl_uncertainties = (
            self.apply_weights3d_same_canonicals_impl(
                nfl_features_processed, weights['w_tensor'], weights['b_tensor']))
        fl_coords2d, fl_coords3d, fl_uncertainties = (
            self.apply_weights3d_same_canonicals_impl(
                fl_features_processed, weights['w_tensor_flipped'], weights['b_tensor_flipped']))
        coords2d = ptu.dynamic_stitch(partitioned_indices, [nfl_coords2d, fl_coords2d])
        coords3d = ptu.dynamic_stitch(partitioned_indices, [nfl_coords3d, fl_coords3d])
        uncertainties = ptu.dynamic_stitch(
            partitioned_indices, [nfl_uncertainties, fl_uncertainties])

        coords2d = model_util.heatmap_to_image(
            coords2d, self.proc_side, self.stride_test, self.centered_stride)
        coords3d = model_util.heatmap_to_metric(
            coords3d, self.proc_side, self.stride_test, self.centered_stride,
            self.box_size_m)
        return coords2d, coords3d, uncertainties

    @torch.jit.export
    def reconstruct_absolute(
            self, coords2d: torch.Tensor, coords3d: torch.Tensor, uncertainties: torch.Tensor,
            intrinsic_matrix: torch.Tensor):
        coords3d_abs = ptu3d.reconstruct_absolute(
            coords2d, coords3d, intrinsic_matrix,
            proc_side=self.proc_side, stride=self.stride_test,
            centered_stride=self.centered_stride,
            weak_perspective=self.weak_perspective,
            mix_3d_inside_fov=0.5,
            point_validity_mask=uncertainties < 0.3,
            border_factor1=1.0, border_factor2=0.6,
            mix_based_on_3d=True) * 1000
        factor = 1 if self.fix_uncert_factor else 3
        return coords3d_abs, uncertainties * factor


def is_hand_joint(name):
    n = name.split('_')[0]
    if any(x in n for x in ['thumb', 'index', 'middle', 'ring', 'pinky']):
        return True

    return (
            (n.startswith('lhan') or n.startswith('rhan')) and len(n) > 4)
