# Copyright (c) OpenMMLab. All rights reserved.
from typing import Optional, Tuple

import torch
import os
from mmcv.cnn import build_norm_layer
from mmcv.ops import DynamicScatter
from torch import Tensor, nn

from mmdet3d.registry import MODELS
from .utils import PFNLayer, get_paddings_indicator
import torch_scatter

@MODELS.register_module()
class PillarFeatureNet(nn.Module):
    """Pillar Feature Net.

    The network prepares the pillar features and performs forward pass
    through PFNLayers.

    Args:
        in_channels (int, optional): Number of input features,
            either x, y, z or x, y, z, r. Defaults to 4.
        feat_channels (tuple, optional): Number of features in each of the
            N PFNLayers. Defaults to (64, ).
        with_distance (bool, optional): Whether to include Euclidean distance
            to points. Defaults to False.
        with_cluster_center (bool, optional): [description]. Defaults to True.
        with_voxel_center (bool, optional): [description]. Defaults to True.
        voxel_size (tuple[float], optional): Size of voxels, only utilize x
            and y size. Defaults to (0.2, 0.2, 4).
        point_cloud_range (tuple[float], optional): Point cloud range, only
            utilizes x and y min. Defaults to (0, -40, -3, 70.4, 40, 1).
        norm_cfg ([type], optional): [description].
            Defaults to dict(type='BN1d', eps=1e-3, momentum=0.01).
        mode (str, optional): The mode to gather point features. Options are
            'max' or 'avg'. Defaults to 'max'.
        legacy (bool, optional): Whether to use the new behavior or
            the original behavior. Defaults to True.
    """

    def __init__(self,
                 in_channels: Optional[int] = 4,
                 feat_channels: Optional[tuple] = (64, ),
                 with_distance: Optional[bool] = False,
                 with_cluster_center: Optional[bool] = True,
                 with_voxel_center: Optional[bool] = True,
                 voxel_size: Optional[Tuple[float]] = (0.2, 0.2, 4),
                 point_cloud_range: Optional[Tuple[float]] = (0, -40, -3, 70.4,
                                                              40, 1),
                 norm_cfg: Optional[dict] = dict(
                     type='BN1d', eps=1e-3, momentum=0.01),
                 mode: Optional[str] = 'max',
                 legacy: Optional[bool] = True):
        super(PillarFeatureNet, self).__init__()
        assert len(feat_channels) > 0
        self.legacy = legacy
        if with_cluster_center:
            in_channels += 3
        if with_voxel_center:
            in_channels += 3
        if with_distance:
            in_channels += 1
        self._with_distance = with_distance
        self._with_cluster_center = with_cluster_center
        self._with_voxel_center = with_voxel_center
        # Create PillarFeatureNet layers
        self.in_channels = in_channels
        feat_channels = [in_channels] + list(feat_channels)
        pfn_layers = []
        for i in range(len(feat_channels) - 1):
            in_filters = feat_channels[i]
            out_filters = feat_channels[i + 1]
            if i < len(feat_channels) - 2:
                last_layer = False
            else:
                last_layer = True
            pfn_layers.append(
                PFNLayer(
                    in_filters,
                    out_filters,
                    norm_cfg=norm_cfg,
                    last_layer=last_layer,
                    mode=mode))
        self.pfn_layers = nn.ModuleList(pfn_layers)

        # Need pillar (voxel) size and x/y offset in order to calculate offset
        self.vx = voxel_size[0]
        self.vy = voxel_size[1]
        self.vz = voxel_size[2]
        self.x_offset = self.vx / 2 + point_cloud_range[0]
        self.y_offset = self.vy / 2 + point_cloud_range[1]
        self.z_offset = self.vz / 2 + point_cloud_range[2]
        self.point_cloud_range = point_cloud_range

    def forward(self, features: Tensor, num_points: Tensor, coors: Tensor,
                *args, **kwargs) -> Tensor:
        """Forward function.

        Args:
            features (torch.Tensor): Point features or raw points in shape
                (N, M, C).
            num_points (torch.Tensor): Number of points in each pillar.
            coors (torch.Tensor): Coordinates of each voxel.

        Returns:
            torch.Tensor: Features of pillars.
        """
        features_ls = [features]
        # Find distance of x, y, and z from cluster center
        if self._with_cluster_center:
            points_mean = features[:, :, :3].sum(
                dim=1, keepdim=True) / num_points.type_as(features).view(
                    -1, 1, 1)
            f_cluster = features[:, :, :3] - points_mean
            features_ls.append(f_cluster)

        # Find distance of x, y, and z from pillar center
        dtype = features.dtype
        if self._with_voxel_center:
            if not self.legacy:
                f_center = torch.zeros_like(features[:, :, :3])
                f_center[:, :, 0] = features[:, :, 0] - (
                    coors[:, 3].to(dtype).unsqueeze(1) * self.vx +
                    self.x_offset)
                f_center[:, :, 1] = features[:, :, 1] - (
                    coors[:, 2].to(dtype).unsqueeze(1) * self.vy +
                    self.y_offset)
                f_center[:, :, 2] = features[:, :, 2] - (
                    coors[:, 1].to(dtype).unsqueeze(1) * self.vz +
                    self.z_offset)
            else:
                f_center = features[:, :, :3]
                f_center[:, :, 0] = f_center[:, :, 0] - (
                    coors[:, 3].type_as(features).unsqueeze(1) * self.vx +
                    self.x_offset)
                f_center[:, :, 1] = f_center[:, :, 1] - (
                    coors[:, 2].type_as(features).unsqueeze(1) * self.vy +
                    self.y_offset)
                f_center[:, :, 2] = f_center[:, :, 2] - (
                    coors[:, 1].type_as(features).unsqueeze(1) * self.vz +
                    self.z_offset)
            features_ls.append(f_center)

        if self._with_distance:
            points_dist = torch.norm(features[:, :, :3], 2, 2, keepdim=True)
            features_ls.append(points_dist)

        # Combine together feature decorations
        features = torch.cat(features_ls, dim=-1)
        # The feature decorations were calculated without regard to whether
        # pillar was empty. Need to ensure that
        # empty pillars remain set to zeros.
        voxel_count = features.shape[1]
        mask = get_paddings_indicator(num_points, voxel_count, axis=0)
        mask = torch.unsqueeze(mask, -1).type_as(features)
        features *= mask

        for pfn in self.pfn_layers:
            features = pfn(features, num_points)

        return features.squeeze(1)

@MODELS.register_module()
class DynamicPillarFeatureNet(PillarFeatureNet):
    """Pillar Feature Net using dynamic voxelization.

    The network prepares the pillar features and performs forward pass
    through PFNLayers. The main difference is that it is used for
    dynamic voxels, which contains different number of points inside a voxel
    without limits.

    만약, 결과 값이 올바르게 나오지 않았다면. Default 4 & feat_channel 64로 재 고려 후 조정 요망.

    Args:
        in_channels (int, optional): Number of input features,
            either x, y, z or x, y, z, r. Defaults to 4. 
        feat_channels (tuple, optional): Number of features in each of the
            N PFNLayers. Defaults to (64, ). -> tuple 32
        with_distance (bool, optional): Whether to include Euclidean distance
            to points. Defaults to False. -> 유클리드 거리. config file은 True
        with_cluster_center (bool, optional): [description]. Defaults to True. -> 클러스터 중심
        with_voxel_center (bool, optional): [description]. Defaults to True. -> 복셀 중심
        voxel_size (tuple[float], optional): Size of voxels, only utilize x 
            and y size. Defaults to (0.2, 0.2, 4). -> 기본 채택
        point_cloud_range (tuple[float], optional): Point cloud range, only
            utilizes x and y min. Defaults to (0, -40, -3, 70.4, 40, 1) = (x_min, y_min, z_min, x_max, y_max, z_max)
        norm_cfg ([type], optional): [description].
            Defaults to dict(type='BN1d', eps=1e-3, momentum=0.01). -> 디폴트
        mode (str, optional): The mode to gather point features. Options are
            'max' or 'avg'. Defaults to 'max'. -> 맥스풀링! / 평균 풀링?
        legacy (bool, optional): Whether to use the new behavior or
            the original behavior. Defaults to True. -> 새 동작 / 원 동작? (변수추가인가)
    """

    def __init__(self,
                 in_channels: Optional[int] = 4,
                 feat_channels: Optional[tuple] = (64, ),
                 with_distance: Optional[bool] = False,
                 with_cluster_center: Optional[bool] = True,
                 with_voxel_center: Optional[bool] = True,
                 voxel_size: Optional[Tuple[float]] = (0.2, 0.2, 8),
                 point_cloud_range: Optional[Tuple[float]] = [-51.2, -51.2, -5.0, 51.2, 51.2, 3.0],
                 norm_cfg: Optional[dict] = dict(
                     type='BN1d', eps=1e-3, momentum=0.01),
                 mode: Optional[str] = 'max',
                 legacy: Optional[bool] = True):
        super(DynamicPillarFeatureNet, self).__init__()    
        # Adjust input channels based on cluster and voxel centers, and distance
        if with_cluster_center:
            in_channels += 3
        if with_voxel_center:
            in_channels += 3
        if with_distance:
            in_channels += 1

        self._with_distance = with_distance
        self._with_cluster_center = with_cluster_center
        self._with_voxel_center = with_voxel_center
        self.voxel_size = voxel_size
        self.point_cloud_range = point_cloud_range
        self.legacy = legacy
        # PFN Layers for feature processing
        feat_channels = [in_channels] + list(feat_channels)
        pfn_layers = []
        for i in range(len(feat_channels) - 1):
            in_filters = feat_channels[i]
            out_filters = feat_channels[i + 1]
            if i > 0:
                in_filters *= 2  # Double channels after first layer
           
            _, norm_layer = build_norm_layer(norm_cfg, out_filters)
            pfn_layers.append(
                nn.Sequential(
                    nn.Linear(in_filters, out_filters, bias=False),
                    norm_layer,
                    nn.ReLU(inplace=True)))
        self.pfn_layers = nn.ModuleList(pfn_layers)

        self.dynamic_scatter = DynamicScatter(voxel_size, point_cloud_range, (mode != 'max'))
        self.cluster_scatter = DynamicScatter(voxel_size, point_cloud_range, average_points=True)

    def forward(self, features: Tensor, coors: Tensor) -> Tuple[Tensor, Tensor]:
        """Forward function for dynamic feature processing.

        Args:
            features (torch.Tensor): Point features or raw points in shape (N, M, C).
            coors (torch.Tensor): Coordinates of each voxel.

        Returns:
            torch.Tensor: Features of pillars.
        """
        features_ls = [features]

        # Cluster center feature calculation
        if self._with_cluster_center:
            voxel_mean, mean_coors = self.cluster_scatter(features, coors)
            points_mean = self.map_voxel_center_to_point(coors, voxel_mean, mean_coors)
            f_cluster = features[:, :, :3] - points_mean[:, :3].view(points_mean.size(0), 1, -1)
            features_ls.append(f_cluster)

        # Voxel center feature calculation
        if self._with_voxel_center:
            f_center = self.calculate_voxel_center(features, coors)
            features_ls.append(f_center)

        # Euclidean distance feature
        if self._with_distance:
            points_dist = torch.norm(features[:, :, :3], 2, 2, keepdim=True)
            features_ls.append(points_dist)

        # Combine features
        features = torch.cat(features_ls, dim=-1)

        for i, pfn in enumerate(self.pfn_layers):
            point_feats = pfn(features)
            voxel_feats, voxel_coors = self.dynamic_scatter(point_feats, coors)

            if i != len(self.pfn_layers) - 1:
                # Append feature back to the next PFN layer
                feat_per_point = self.map_voxel_center_to_point(coors, voxel_feats, voxel_coors)
                features = torch.cat([point_feats, feat_per_point], dim=1)

        return voxel_feats, voxel_coors

    def map_voxel_center_to_point(self, pts_coors: Tensor, voxel_mean: Tensor,
                                  voxel_coors: Tensor) -> Tensor:
        """Map voxel centers to corresponding points.

        Args:
            pts_coors (torch.Tensor): Point coordinates in (M, 3).
            voxel_mean (torch.Tensor): Aggregated features of a voxel in (N, C).
            voxel_coors (torch.Tensor): Voxel coordinates.

        Returns:
            torch.Tensor: Voxel center features for each point in (M, C).
        """
        # Step 1: Scatter voxel means into a canvas (x, y)
        canvas_y = int((self.point_cloud_range[4] - self.point_cloud_range[1]) / self.voxel_size[1])
        canvas_x = int((self.point_cloud_range[3] - self.point_cloud_range[0]) / self.voxel_size[0])
        canvas_channel = voxel_mean.size(1)
        batch_size = pts_coors[-1, 0] + 1
        canvas_len = canvas_y * canvas_x * batch_size

        canvas = voxel_mean.new_zeros(canvas_channel, canvas_len)
        indices = (
            voxel_coors[:, 0] * canvas_y * canvas_x +
            voxel_coors[:, 2] * canvas_x + voxel_coors[:, 3])

        canvas[:, indices.long()] = voxel_mean.t()

        # Step 2: Extract voxel features for each point
        voxel_index = (
            pts_coors[:, 0] * canvas_y * canvas_x +
            pts_coors[:, 2] * canvas_x + pts_coors[:, 3])
        center_per_point = canvas[:, voxel_index.long()].t()
        return center_per_point

    def calculate_voxel_center(self, features: Tensor, coors: Tensor) -> Tensor:
        """Calculate the voxel center for each feature.

        Args:
            features (torch.Tensor): Input features.
            coors (torch.Tensor): Voxel coordinates.

        Returns:
            torch.Tensor: Calculated voxel center features.
        """
        f_center = torch.zeros_like(features[:, :, :3])
        f_center[:, :, 0] = features[:, :, 0] - (coors[:, 3].type_as(features).unsqueeze(1) * self.voxel_size[0] + (self.voxel_size[0] / 2))
        f_center[:, :, 1] = features[:, :, 1] - (coors[:, 2].type_as(features).unsqueeze(1) * self.voxel_size[1] + (self.voxel_size[1] / 2))
        f_center[:, :, 2] = features[:, :, 2] - (coors[:, 1].type_as(features).unsqueeze(1) * self.voxel_size[2] + (self.voxel_size[2] / 2))
        return f_center

#SAV(Pillar) OK
@MODELS.register_module()
class SelfAdaptivePillarization(nn.Module):
    """Self-Adaptive Pillar Feature Net with adaptive voxelization in x and y axes.

    이 네트워크는 동적 보셀화를 사용하여 포인트 클라우드 데이터를 처리하며,
    x와 y 축에 대해 보셀 크기를 포인트 밀도와 분포에 따라 동적으로 조절합니다.
    z축의 보셀 크기는 고정되어 있습니다.

    Args:
        in_channels (int, optional): 입력 특징의 채널 수. 기본값은 4.
        feat_channels (tuple, optional): PFN 레이어의 채널 수. 기본값은 (64, ).
        with_distance (bool, optional): 유클리드 거리 사용 여부. 기본값은 False.
        with_cluster_center (bool, optional): 클러스터 중심 사용 여부. 기본값은 True.
        with_voxel_center (bool, optional): 보셀 중심 사용 여부. 기본값은 True.
        voxel_size (tuple[float], optional): 보셀 크기. 기본값은 (0.2, 0.2, 4).
        point_cloud_range (tuple[float], optional): 포인트 클라우드 범위.
        norm_cfg (dict, optional): 정규화 레이어 설정. 기본값은 BN1d.
        mode (str, optional): 포인트 특징 집계 모드('max' 또는 'avg'). 기본값은 'max'.
        alpha (float, optional): 스케일링 팩터 계산을 위한 가중치. 기본값은 0.5.
    """

    def __init__(self,
                 in_channels: int = 4,
                 feat_channels: tuple = (64,),
                 with_distance: bool = False,
                 with_cluster_center: bool = True,
                 with_voxel_center: bool = True,
                 voxel_size: Tuple[float, float, float] = (0.2, 0.2, 4),
                 point_cloud_range: Tuple[float, float, float, float, float, float] = (-51.2, -51.2, -5.0, 51.2, 51.2, 3.0),
                 norm_cfg: dict = dict(type='BN1d', eps=1e-3, momentum=0.01),
                 mode: str = 'max',
                 alpha: float = 0.5,
                 legacy=None):
        super(SelfAdaptivePillarization, self).__init__()
        self.alpha = alpha
        self.with_distance = with_distance
        self.with_cluster_center = with_cluster_center
        self.with_voxel_center = with_voxel_center
        self.voxel_size = voxel_size
        self.point_cloud_range = point_cloud_range
        self.legacy = legacy


        # 추가 채널 계산
        additional_channels = 0
        if self.with_cluster_center:
            additional_channels += 3
        if self.with_voxel_center:
            additional_channels += 3
        if self.with_distance:
            additional_channels += 1

        # PFN 레이어 구성
        feat_channels = [in_channels + additional_channels] + list(feat_channels)
        pfn_layers = []
        for i in range(len(feat_channels) - 1):
            in_filters = feat_channels[i]
            out_filters = feat_channels[i + 1]
            if i > 0:
                in_filters *= 2  # 첫 레이어 이후에는 채널 수를 두 배로
            _, norm_layer = build_norm_layer(norm_cfg, out_filters)
            pfn_layers.append(
                nn.Sequential(
                    nn.Linear(in_filters, out_filters, bias=False),
                    norm_layer,
                    nn.ReLU(inplace=True)))
        self.pfn_layers = nn.ModuleList(pfn_layers)

        # DynamicScatter 초기화 (adaptive voxelization 적용)
        self.dynamic_scatter = DynamicScatter(self.voxel_size, self.point_cloud_range, (mode != 'max'))
        self.cluster_scatter = DynamicScatter(self.voxel_size, self.point_cloud_range, average_points=True)
        
        # Config 값 기반으로 distance_max, density_max 계산
        self.distance_max, self.density_max = self.calculate_max_values()

        # 가중치 초기화 - Kaiming Normal 사용
        for pfn in self.pfn_layers:
            nn.init.kaiming_normal_(pfn[0].weight, nonlinearity='relu')
    
    def calculate_max_values(self):
        """data_preprocessor에서 max_num_points 값을 가져와 distance_max와 density_max 계산."""
        # data_preprocessor에서 max_num_points 가져오기
        max_num_points = 32
        
        # Voxel 크기와 밀도를 기반으로 최대 값 계산
        voxel_size_tensor = torch.tensor(self.voxel_size, dtype=torch.float32)
        distance_max = torch.norm(voxel_size_tensor).item()  # Voxel 내 최대 거리
        density_max = max_num_points  # Voxel 내 최대 밀도 = 최대 점
        
        return distance_max, density_max
    
    def calculate_adaptive_voxel_size(self, features, coors):
        """포인트 밀도와 거리 기반으로 x, y 축의 보셀 크기를 동적으로 계산합니다."""
        with torch.no_grad():
            # 밀도=점, 상대위치= 점평균 - 중심 
            _, inverse_indices, counts = torch.unique(coors, return_inverse=True, return_counts=True, dim=0)
            counts = counts.float()
            point_density = counts[inverse_indices]

            voxel_size_tensor = torch.tensor(self.voxel_size, device=features.device, dtype=features.dtype)
            voxel_centers = coors[:, 1:].float() * voxel_size_tensor     
            points_mean = features[:, :, :3].mean(dim=1)  # (N, 3)
            point_distance = torch.norm(points_mean - voxel_centers, dim=1)  # (N,)
            
            # 정규화: distance_max와 density_max 기반으로 정규화
            normalized_point_distance = point_distance / self.distance_max
            normalized_point_density = point_density / self.density_max
            
            # 스케일링 팩터 계산
            scaling_factor = torch.clamp(
                torch.log1p(
                    self.alpha * torch.sqrt(normalized_point_distance + 1e-6) +
                    (1 - self.alpha) * (1 / torch.sqrt(normalized_point_density + 1e-6))
                ),
                min=0.5, max=1.5  # 스케일링 범위 조정 가능.
            )  # (N,)

            # x와 y 축에만 스케일링 적용, z축은 고정
            adaptive_voxel_size = torch.stack([
                self.voxel_size[0] * scaling_factor,  # x축 스케일링
                self.voxel_size[1] * scaling_factor,  # y축 스케일링
                torch.full_like(scaling_factor, self.voxel_size[2])  # z축은 고정
            ], dim=1)  # (N, 3)
            return adaptive_voxel_size

    def forward(self, features: Tensor, num_points: Tensor, coors: Tensor) -> Tensor:
        """전방 전달 함수로, Self-Adaptive Voxelization을 적용합니다."""
        # 적응형 보셀 크기 계산
        adaptive_voxel_size = self.calculate_adaptive_voxel_size(features, coors)

        self.dynamic_scatter.voxel_size = adaptive_voxel_size
        self.cluster_scatter.voxel_size = adaptive_voxel_size

        features_ls = [features]

        # 클러스터 중심 특징 계산
        if self.with_cluster_center:
            voxel_mean, mean_coors = self.cluster_scatter(features, coors)
            points_mean = self.map_voxel_center_to_point(coors, voxel_mean, mean_coors)
            f_cluster = features[:, :, :3] - points_mean[:, :3].view(points_mean.size(0), 1, -1)
            features_ls.append(f_cluster)

        # 보셀 중심 특징 계산
        if self.with_voxel_center:
            f_center = self.calculate_voxel_center(features, coors)
            features_ls.append(f_center)

        # 유클리드 거리 특징 계산
        if self.with_distance:
            points_dist = torch.norm(features[:, :, :3], dim=2, keepdim=True)
            features_ls.append(points_dist)

        # 특징 결합
        features = torch.cat(features_ls, dim=-1)
        voxel_count = features.shape[1]
        mask = get_paddings_indicator(num_points, voxel_count, axis=0)
        mask = torch.unsqueeze(mask, -1).type_as(features)
        features *= mask

        for i, pfn in enumerate(self.pfn_layers):
            point_feats = pfn(features)
            voxel_feats, _ = self.dynamic_scatter(point_feats, coors)

            if i != len(self.pfn_layers) - 1:
                # 다음 레이어를 위해 보셀 특징을 포인트 특징에 맵핑
                feat_per_point = self.map_voxel_center_to_point(coors, voxel_feats, voxel_coors)
                features = torch.cat([point_feats, feat_per_point], dim=-1)

        return voxel_feats #, voxel_coors

    def map_voxel_center_to_point(self, pts_coors: Tensor, voxel_mean: Tensor,
                                  voxel_coors: Tensor) -> Tensor:
        """보셀 중심 특징을 각 포인트에 매핑합니다."""
        # 캔버스 크기 계산
        canvas_y = int((self.point_cloud_range[4] - self.point_cloud_range[1]) / self.voxel_size[1])
        canvas_x = int((self.point_cloud_range[3] - self.point_cloud_range[0]) / self.voxel_size[0])
        canvas_channel = voxel_mean.size(1)
        batch_size = int(pts_coors[:, 0].max().item()) + 1
        canvas_len = canvas_y * canvas_x * batch_size

        canvas = voxel_mean.new_zeros(canvas_channel, canvas_len)
        indices = (
            voxel_coors[:, 0] * canvas_y * canvas_x +
            voxel_coors[:, 2] * canvas_x + voxel_coors[:, 3]
        ).long()

        canvas[:, indices] = voxel_mean.t()

        voxel_index = (
            pts_coors[:, 0] * canvas_y * canvas_x +
            pts_coors[:, 2] * canvas_x + pts_coors[:, 3]
        ).long()
        center_per_point = canvas[:, voxel_index].t()
        return center_per_point

    def calculate_voxel_center(self, features: Tensor, coors: Tensor) -> Tensor:
        """각 포인트에 대한 보셀 중심과의 상대 위치를 계산합니다."""
        f_center = torch.zeros_like(features[:, :, :3])
        voxel_size_tensor = torch.tensor(self.voxel_size, device=features.device, dtype=features.dtype)
        # 보셀 좌표를 실수형으로 변환하여 보셀 중심 계산
        voxel_center = (coors[:, 1:].float() + 0.5) * voxel_size_tensor
        f_center = features[:, :, :3] - voxel_center.unsqueeze(1)
        return f_center

#SAV-Dynamic (Pillar) OK
@MODELS.register_module()
class SelfAdaptivePillarization_D(nn.Module):
    """Self-Adaptive Pillar Feature Net with key adaptive voxelization features."""
    # Dynamic 요소에 필요한 모델에 적용

    def __init__(self,
                 in_channels: int = 4,
                 feat_channels: tuple = (64,),
                 with_distance: bool = False,
                 with_cluster_center: bool = True,
                 with_voxel_center: bool = True,
                 voxel_size: Tuple[float, float, float] = (0.2, 0.2, 4),
                 point_cloud_range: Tuple[float, float, float, float, float, float] = (-51.2, -51.2, -5.0, 51.2, 51.2, 3.0),
                 norm_cfg: dict = dict(type='BN1d', eps=1e-3, momentum=0.01),
                 mode: str = 'max',
                 alpha: float = 0.5,
                 legacy=None):
        super(SelfAdaptivePillarization_D, self).__init__()
        self.alpha = alpha
        self.with_distance = with_distance
        self.with_cluster_center = with_cluster_center
        self.with_voxel_center = with_voxel_center
        self.voxel_size = voxel_size
        self.point_cloud_range = point_cloud_range
        self.legacy = legacy
        
        # 추가 채널 계산
        additional_channels = 0
        if self.with_cluster_center:
            additional_channels += 3
        if self.with_voxel_center:
            additional_channels += 3
        if self.with_distance:
            additional_channels += 1

        # PFN 레이어 구성
        feat_channels = [in_channels + additional_channels] + list(feat_channels)
        pfn_layers = []
        for i in range(len(feat_channels) - 1):
            in_filters = feat_channels[i]
            out_filters = feat_channels[i + 1]
            if i > 0:
                in_filters *= 2  # 첫 레이어 이후에는 채널 수를 두 배로
            _, norm_layer = build_norm_layer(norm_cfg, out_filters)
            pfn_layers.append(
                nn.Sequential(
                    nn.Linear(in_filters, out_filters, bias=False),
                    norm_layer,
                    nn.ReLU(inplace=True)))
        self.pfn_layers = nn.ModuleList(pfn_layers)

        # DynamicScatter 초기화 (adaptive voxelization 적용)
        self.dynamic_scatter = DynamicScatter(self.voxel_size, self.point_cloud_range, (mode != 'max'))
        self.cluster_scatter = DynamicScatter(self.voxel_size, self.point_cloud_range, average_points=True)
        
        # Config 값 기반으로 distance_max, density_max 계산
        self.distance_max, self.density_max = self.calculate_max_values()

        # 가중치 초기화 - Kaiming Normal 사용
        for pfn in self.pfn_layers:
            nn.init.kaiming_normal_(pfn[0].weight, nonlinearity='relu')
    
    def calculate_max_values(self):
        """data_preprocessor에서 max_num_points 값을 가져와 distance_max와 density_max 계산."""
        # data_preprocessor에서 max_num_points 가져오기
        max_num_points = 32
        
        # Voxel 크기와 밀도를 기반으로 최대 값 계산
        voxel_size_tensor = torch.tensor(self.voxel_size, dtype=torch.float32)
        distance_max = torch.norm(voxel_size_tensor).item()  # Voxel 내 최대 거리
        density_max = max_num_points  # Voxel 내 최대 밀도 = 최대 점
        
        return distance_max, density_max

    def calculate_adaptive_voxel_size(self, features, coors):
        """포인트 밀도와 거리 기반으로 x, y 축의 보셀 크기를 동적으로 계산합니다."""
        with torch.no_grad():
            # 밀도=점, 상대위치= 점평균 - 중심 
            _, inverse_indices, counts = torch.unique(coors, return_inverse=True, return_counts=True, dim=0)
            counts = counts.float()
            point_density = counts[inverse_indices]

            voxel_size_tensor = torch.tensor(self.voxel_size, device=features.device, dtype=features.dtype)
            voxel_centers = coors[:, 1:].float() * voxel_size_tensor
            points_mean = features[:, :, :3].mean(dim=1)  # (N, 3)
            point_distance = torch.norm(points_mean - voxel_centers, dim=1)  # (N,)
            
            # 정규화: distance_max와 density_max 기반으로 정규화
            normalized_point_distance = point_distance / self.distance_max
            normalized_point_density = point_density / self.density_max

            # 스케일링 팩터 계산
            scaling_factor = torch.clamp(
                torch.log1p(
                    self.alpha * torch.sqrt(normalized_point_distance + 1e-6) +
                    (1 - self.alpha) * (1 / torch.sqrt(normalized_point_density + 1e-6))
                ),
                min=0.5, max=1.5  # 스케일링 범위 조정 가능.
            )  # (N,)

            # x와 y 축에만 스케일링 적용, z축은 고정
            adaptive_voxel_size = torch.stack([
                self.voxel_size[0] * scaling_factor,  # x축 스케일링
                self.voxel_size[1] * scaling_factor,  # y축 스케일링
                torch.full_like(scaling_factor, self.voxel_size[2])  # z축은 고정
            ], dim=1)  # (N, 3)
            return adaptive_voxel_size
    
    def forward(self, features: Tensor, num_points: Tensor, coors: Tensor) -> Tuple[Tensor, Tensor]:
        # 적응형 보셀 크기 계산
        adaptive_voxel_size = self.calculate_adaptive_voxel_size(features, coors)

        self.dynamic_scatter.voxel_size = adaptive_voxel_size
        self.cluster_scatter.voxel_size = adaptive_voxel_size

        features_ls = [features]

        # 클러스터 중심 특징 계산
        if self.with_cluster_center:
            voxel_mean, mean_coors = self.cluster_scatter(features, coors)
            points_mean = self.map_voxel_center_to_point(coors, voxel_mean, mean_coors)
            f_cluster = features[:, :, :3] - points_mean[:, :3].view(points_mean.size(0), 1, -1)
            features_ls.append(f_cluster)

        # 보셀 중심 특징 계산
        if self.with_voxel_center:
            f_center = self.calculate_voxel_center(features, coors)
            features_ls.append(f_center)

        # 유클리드 거리 특징 계산
        if self.with_distance:
            points_dist = torch.norm(features[:, :, :3], dim=2, keepdim=True)
            features_ls.append(points_dist)

        # 특징 결합
        features = torch.cat(features_ls, dim=-1)
        voxel_count = features.shape[1]
        mask = get_paddings_indicator(num_points, voxel_count, axis=0)
        mask = torch.unsqueeze(mask, -1).type_as(features)
        features *= mask
        
        for i, pfn in enumerate(self.pfn_layers):
            point_feats = pfn(features)
            voxel_feats, voxel_coors = self.dynamic_scatter(point_feats, coors)

            if i != len(self.pfn_layers) - 1:
                # 다음 레이어를 위해 보셀 특징을 포인트 특징에 맵핑
                feat_per_point = self.map_voxel_center_to_point(coors, voxel_feats, voxel_coors)
                features = torch.cat([point_feats, feat_per_point], dim=-1)

        return voxel_feats, voxel_coors

    def map_voxel_center_to_point(self, pts_coors: Tensor, voxel_mean: Tensor,
                                  voxel_coors: Tensor) -> Tensor:
        """보셀 중심 특징을 각 포인트에 매핑합니다."""
        # 캔버스 크기 계산
        canvas_y = int((self.point_cloud_range[4] - self.point_cloud_range[1]) / self.voxel_size[1])
        canvas_x = int((self.point_cloud_range[3] - self.point_cloud_range[0]) / self.voxel_size[0])
        canvas_channel = voxel_mean.size(1)
        batch_size = int(pts_coors[:, 0].max().item()) + 1
        canvas_len = canvas_y * canvas_x * batch_size

        canvas = voxel_mean.new_zeros(canvas_channel, canvas_len)
        indices = (
            voxel_coors[:, 0] * canvas_y * canvas_x +
            voxel_coors[:, 2] * canvas_x + voxel_coors[:, 3]
        ).long()

        canvas[:, indices] = voxel_mean.t()

        voxel_index = (
            pts_coors[:, 0] * canvas_y * canvas_x +
            pts_coors[:, 2] * canvas_x + pts_coors[:, 3]
        ).long()
        center_per_point = canvas[:, voxel_index].t()
        return center_per_point

    def calculate_voxel_center(self, features: Tensor, coors: Tensor) -> Tensor:
        """각 포인트에 대한 보셀 중심과의 상대 위치를 계산합니다."""
        f_center = torch.zeros_like(features[:, :, :3])
        voxel_size_tensor = torch.tensor(self.voxel_size, device=features.device, dtype=features.dtype)
        # 보셀 좌표를 실수형으로 변환하여 보셀 중심 계산
        voxel_center = (coors[:, 1:].float() + 0.5) * voxel_size_tensor
        f_center = features[:, :, :3] - voxel_center.unsqueeze(1)
        return f_center

#SAFE(Pillar) 
@MODELS.register_module()
class SelfAdaptiveFeatureExtractor_P(nn.Module):
    """Pillar Feature Net.

    이 네트워크는 Pillar Feature를 준비하고 PFNLayer를 통해 forward pass를 수행합니다.

    Args:
        in_channels (int, optional): 입력 Feature의 수, x, y, z 또는 x, y, z, r일 수 있습니다. 기본값은 4입니다.
        feat_channels (tuple, optional): N개의 PFNLayer 각각의 Feature 수입니다. 기본값은 (64, ).
        with_distance (bool, optional): 유클리드 거리를 포함할지 여부. 기본값은 False입니다.
        with_cluster_center (bool, optional): 클러스터 중심을 포함할지 여부. 기본값은 True입니다.
        with_voxel_center (bool, optional): Voxel 중심을 포함할지 여부. 기본값은 True입니다.
        voxel_size (tuple[float], optional): Voxel 크기, x와 y 크기만 사용합니다. 기본값은 (0.2, 0.2, 4)입니다.
        point_cloud_range (tuple[float], optional): 포인트 클라우드 범위, x와 y의 최소값만 사용합니다. 기본값은 (0, -40, -3, 70.4, 40, 1)입니다.
        norm_cfg ([type], optional): 정규화 구성. 기본값은 dict(type='BN1d', eps=1e-3, momentum=0.01)입니다.
        mode (str, optional): 포인트 Feature를 모으는 모드입니다. 'max' 또는 'avg'가 가능합니다. 기본값은 'max'입니다.
        legacy (bool, optional): 원래 동작을 사용할지 여부. 기본값은 True입니다.
    """

    def __init__(self,
                 in_channels: Optional[int] = 4,
                 feat_channels: Optional[tuple] = (64, ),
                 with_distance: Optional[bool] = False,
                 with_cluster_center: Optional[bool] = True,
                 with_voxel_center: Optional[bool] = True,
                 voxel_size: Optional[Tuple[float]] = (0.2, 0.2, 4),
                 point_cloud_range: Optional[Tuple[float]] = (0, -40, -3, 70.4,
                                                              40, 1),
                 norm_cfg: Optional[dict] = dict(
                     type='BN1d', eps=1e-3, momentum=0.01),
                 mode: Optional[str] = 'max',
                 legacy: Optional[bool] = True):
        super(SelfAdaptiveFeatureExtractor_P, self).__init__()
        assert len(feat_channels) > 0
        self.legacy = legacy
        if with_cluster_center:
            in_channels += 3
        if with_voxel_center:
            in_channels += 3
        if with_distance:
            in_channels += 1
        self._with_distance = with_distance
        self._with_cluster_center = with_cluster_center
        self._with_voxel_center = with_voxel_center
        # PillarFeatureNet 레이어 생성
        self.in_channels = in_channels
        feat_channels = [in_channels+2] + list(feat_channels)
        pfn_layers = []
        for i in range(len(feat_channels) - 1):
            in_filters = feat_channels[i]
            out_filters = feat_channels[i + 1]
            if i < len(feat_channels) - 2:
                last_layer = False
            else:
                last_layer = True
            pfn_layers.append(
                PFNLayer(
                    in_filters,
                    out_filters,
                    norm_cfg=norm_cfg,
                    last_layer=last_layer,
                    mode=mode))
        self.pfn_layers = nn.ModuleList(pfn_layers)

        # Voxel 크기 및 x/y 오프셋 설정
        self.vx = voxel_size[0]
        self.vy = voxel_size[1]
        self.vz = voxel_size[2]
        self.x_offset = self.vx / 2 + point_cloud_range[0]
        self.y_offset = self.vy / 2 + point_cloud_range[1]
        self.z_offset = self.vz / 2 + point_cloud_range[2]
        self.point_cloud_range = point_cloud_range

    def calculate_spatial_features(self, features: Tensor, coors: Tensor) -> Tensor:
        """공간적 Feature(점 밀도 및 거리)를 계산합니다.

        Args:
            features (torch.Tensor): (N, C) 형태의 점 Feature.
            coors (torch.Tensor): (N, 4) 형태의 좌표 정보.

        Returns:
            torch.Tensor: Voxel에 추가할 공간적 Feature. (N, 2)
        """    
        # 밀도=점/복셀, 상대위치= (점-중심)/평균
        unique_voxels, inverse_indices, counts = torch.unique(coors, return_inverse=True, return_counts=True, dim=0)
        safe_num_points = torch.clamp(counts.float(), min=1)  # (V,)
        point_density = safe_num_points / (self.vx * self.vy * self.vz)  # (V,)
        
        sum_coords = torch.zeros((unique_voxels.size(0), 3), device=features.device, dtype=features.dtype)
        sum_coords = sum_coords.index_add_(0, inverse_indices, features[:, :3])  # (V, 3)
        voxel_mean = sum_coords / safe_num_points.unsqueeze(1)  # (V, 3)
        point_mean = voxel_mean[inverse_indices]  # (N, 3)
        point_distances = torch.norm(features[:, :3] - point_mean, dim=1)  # (N,)

        sum_distances = torch.zeros((unique_voxels.size(0),), device=features.device, dtype=features.dtype)
        sum_distances = sum_distances.index_add_(0, inverse_indices, point_distances)  # (V,)
        voxel_mean_distance = sum_distances / safe_num_points  # (V,)
        mean_distance = voxel_mean_distance[inverse_indices]  # (N,)

        f_spatial = torch.cat([point_density[inverse_indices].unsqueeze(1), mean_distance.unsqueeze(1)], dim=1)  # (N, 2)

        return f_spatial


    def forward(self, features: torch.Tensor, num_points: torch.Tensor, coors: torch.Tensor,
                *args, **kwargs) -> torch.Tensor:
        """Self-Adaptive Voxelization을 반영한 forward 함수.

        Args:
            features (torch.Tensor): (N, M, C) 형태의 점 Feature.
            num_points (torch.Tensor): 각 Pillar 내의 점 개수.
            coors (torch.Tensor): 각 Voxel의 좌표.

        Returns:
            torch.Tensor: Pillar의 Feature.
        """
        features_ls = [features]

        # 클러스터 중심으로부터의 거리 계산
        if self._with_cluster_center:
            points_mean = features[:, :, :3].sum(
                dim=1, keepdim=True) / num_points.type_as(features).view(
                    -1, 1, 1)
            f_cluster = features[:, :, :3] - points_mean
            features_ls.append(f_cluster)

        # Pillar 중심으로부터의 거리 계산
        dtype = features.dtype
        if self._with_voxel_center:
            if not self.legacy:
                f_center = torch.zeros_like(features[:, :, :3])
                f_center[:, :, 0] = features[:, :, 0] - (
                    coors[:, 3].to(dtype).unsqueeze(1) * self.vx +
                    self.x_offset)
                f_center[:, :, 1] = features[:, :, 1] - (
                    coors[:, 2].to(dtype).unsqueeze(1) * self.vy +
                    self.y_offset)
                f_center[:, :, 2] = features[:, :, 2] - (
                    coors[:, 1].to(dtype).unsqueeze(1) * self.vz +
                    self.z_offset)
            else:
                f_center = features[:, :, :3]
                f_center[:, :, 0] = f_center[:, :, 0] - (
                    coors[:, 3].type_as(features).unsqueeze(1) * self.vx +
                    self.x_offset)
                f_center[:, :, 1] = f_center[:, :, 1] - (
                    coors[:, 2].type_as(features).unsqueeze(1) * self.vy +
                    self.y_offset)
                f_center[:, :, 2] = f_center[:, :, 2] - (
                    coors[:, 1].type_as(features).unsqueeze(1) * self.vz +
                    self.z_offset)
            features_ls.append(f_center)

        if self._with_distance:
            points_dist = torch.norm(features[:, :, :3], 2, 2, keepdim=True)
            features_ls.append(points_dist)

        # 유효한 포인트를 위한 마스크 생성
        voxel_count = features.shape[1]
        mask = get_paddings_indicator(num_points, voxel_count, axis=0)
        mask = mask.type(torch.bool)  # 불리언 마스크로 변환

    # features와 coors 평탄화
        features_unpadded = features[mask]  # 형태: (N_points, C)
        coors_per_point = coors.unsqueeze(1).expand(-1, voxel_count, -1)[mask]  # 형태: (N_points, 4)
      
        f_spatial_unpadded = self.calculate_spatial_features(features_unpadded, coors_per_point)  # 형태: (N_points, 2)
        f_spatial = torch.zeros((features.size(0), features.size(1), f_spatial_unpadded.size(-1)), device=features.device)
        f_spatial[mask] = f_spatial_unpadded

    # 공간적 특징을 features 리스트에 추가
        features_ls.append(f_spatial)
        features = torch.cat(features_ls, dim=-1)
        mask = mask.unsqueeze(-1).type_as(features)
        features *= mask

        # VFE 레이어를 통과
        for vfe in self.pfn_layers:
            if features.dim() == 2:
                features = features.unsqueeze(2)  # 필요시 3D로 조정
            features = vfe(features)

        return features.squeeze(1)
    
    def map_voxel_center_to_point(self, pts_coors: Tensor, voxel_mean: Tensor,
                                  voxel_coors: Tensor) -> Tensor:
        """보셀 중심 특징을 각 포인트에 매핑합니다."""
        # 캔버스 크기 계산
        canvas_y = int((self.point_cloud_range[4] - self.point_cloud_range[1]) / self.voxel_size[1])
        canvas_x = int((self.point_cloud_range[3] - self.point_cloud_range[0]) / self.voxel_size[0])
        canvas_channel = voxel_mean.size(1)
        batch_size = int(pts_coors[:, 0].max().item()) + 1
        canvas_len = canvas_y * canvas_x * batch_size

        canvas = voxel_mean.new_zeros(canvas_channel, canvas_len)
        indices = (
            voxel_coors[:, 0] * canvas_y * canvas_x +
            voxel_coors[:, 2] * canvas_x + voxel_coors[:, 3]
        ).long()

        canvas[:, indices] = voxel_mean.t()

        voxel_index = (
            pts_coors[:, 0] * canvas_y * canvas_x +
            pts_coors[:, 2] * canvas_x + pts_coors[:, 3]
        ).long()
        center_per_point = canvas[:, voxel_index].t()
        return center_per_point

    def calculate_voxel_center(self, features: Tensor, coors: Tensor) -> Tensor:
        """각 포인트에 대한 보셀 중심과의 상대 위치를 계산합니다."""
        f_center = torch.zeros_like(features[:, :, :3])
        voxel_size_tensor = torch.tensor(self.voxel_size, device=features.device, dtype=features.dtype)
        # 보셀 좌표를 실수형으로 변환하여 보셀 중심 계산
        voxel_center = (coors[:, 1:].to(features.dtype) + 0.5) * voxel_size_tensor
        f_center = features[:, :, :3] - voxel_center.unsqueeze(1)
        return f_center


    