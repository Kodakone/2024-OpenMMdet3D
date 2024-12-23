# 0625 작업하기.,  openmmlab에 맞춘 ver.
# 위 모델은 시험용으로 제작되었음

voxel_size = [0.2, 0.2, 4]

# model settings
model = dict(
    type="PillarNet",
    # data Partition ~ Grouping? 그대로 가져감
    data_preprocessor=dict(
        type='Det3DDataPreprocessor',
        voxel=True,
        voxel_layer=dict(
            max_num_points=32,
            voxel_size=voxel_size,
            max_voxels=(30000, 40000))),
    
    # Voxel In channel -> (x,y,z,r,t)
    voxel_encoder=dict(
        type='SelfAdaptiveFeatureNet_Dynamic',
        in_channels=5,
        feat_channels=(32, ),
        with_distance=False,
        voxel_size=voxel_size,
        norm_cfg=dict(type='BN1d', eps=1e-3, momentum=0.01),
        legacy=False),
    middle_encoder=dict(
        type='PointPillarsScatter', in_channels=32, output_shape=(512, 512)),

    # Backbone
    backbone=dict(
        type='Pillar2DCNN',
        in_channels=32,
        out_channels=[32, 64, 128, 256],
        # 4개 사용. 32, 64, 128, 256
        norm_cfg=dict(type='BN1d', eps=1e-3, momentum=0.01),
       ),
   
    # 아웃 채널 전
    neck=dict(
        type='PillarConcat',
        in_channels=256,
        out_channels=512,
        norm_cfg=dict(type='BN', eps=1e-3, momentum=0.01),
        upsample_cfg=dict(type='deconv', bias=False),
        conv_cfg=dict(type='Conv2d', bias=False),
        ),
    
    # bbox_head
    bbox_head=dict(
        type="CenterHead_PillarNet",
        in_channels=512,
        tasks=[
            dict(num_class=1, class_names=['car']),
            dict(num_class=2, class_names=['truck', 'construction_vehicle']),
            dict(num_class=2, class_names=['bus', 'trailer']),
            dict(num_class=1, class_names=['barrier']),
            dict(num_class=2, class_names=['motorcycle', 'bicycle']),
            dict(num_class=2, class_names=['pedestrian', 'traffic_cone']),
        ],
        common_heads=dict(
            reg=(2, 2), height=(1, 2), dim=(3, 2), rot=(2, 4), vel=(2, 4)),
            # reg=(2, 2), height=(1, 2), dim=(3, 2), rot=(2, 2), vel=(2, 2)
        share_conv_channel=64,
        bbox_coder=dict(
            type='CenterPointBBoxCoder',
            post_center_range = [-51.2, -51.2, -5.0, 51.2, 51.2, 3.0],
            # [-61.2, -61.2, -10.0, 61.2, 61.2, 10.0]
            max_num=500,
            score_threshold=0.3, # 0.1
            out_size_factor=8,
            voxel_size=[0.2, 0.2],
            code_size=9),
        separate_head=dict(
            type='SeparateHead', init_bias=-2.19, final_kernel=3),
        loss_cls=dict(type='mmdet.GaussianFocalLoss', reduction='mean'),
        loss_bbox=dict(
            type='mmdet.L1Loss', reduction='mean', loss_weight=0.25), # L1Loss
        norm_bbox=True),
    

    train_cfg = dict(
        grid_size=[512, 512, 1],
        voxel_size=[0.2, 0.2, 4], 
        out_size_factor=8,
        dense_reg=1,
        gaussian_overlap=0.1,
        max_objs=500,
        min_radius=2,  # 원본 2
        code_weights=[1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.5, 0.5],
        #[1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.2, 0.2]
        point_cloud_range = [-51.2, -51.2, -5.0, 51.2, 51.2, 3.0] 
        # [-61.2, -61.2, -10.0, 61.2, 61.2, 10.0]
    ),


    test_cfg = dict(
        post_center_limit_range=[-61.2, -61.2, -10.0, 61.2, 61.2, 10.0],
            max_per_img=300, #다른 500은 건들 ㄴㄴ, 500 
            max_pool_nms=False,
            min_radius=[4, 12, 10, 1, 0.85, 0.175],
            score_threshold=0.3, # 0.1
            out_size_factor=8,
            voxel_size=[0.2, 0.2],
            nms_type='rotate',
            pre_max_size=1000,
            post_max_size=50, # 83
            nms_thr=0.1)) # 0.2