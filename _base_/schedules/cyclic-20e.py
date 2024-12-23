# For nuScenes dataset, we usually evaluate the model at the end of training.
# Since the models are trained by 24 epochs by default, we set evaluation
# interval to be 20. Please change the interval accordingly if you do not
# use a default schedule.
# optimizer
lr = 1e-4
# This schedule is mainly used by models on nuScenes dataset
# max_norm=10 is better for SECOND
optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(type='AdamW', lr=lr, weight_decay=0.01),
    clip_grad=dict(max_norm=35, norm_type=2))
# learning rate
param_scheduler = [
    # learning rate scheduler, 
    # During the first 8 epochs, learning rate increases from 0 to lr * 10 (1e-3)
    # during the next 12 epochs, learning rate decreases from lr * 10 (1e-3) to lr * 1e-4 (1e-8)
    # 0906에, 학습률 구간 및 lr 등 다수 조정 있었음.(0~4, 4~16)
# 뉴씬 쓸거면 10 ㄱ
    dict(
        type='CosineAnnealingLR',
        T_max=4,
        eta_min=lr * 10,
        begin=0,
        end=4,
        by_epoch=True,
        convert_to_iter_based=True),
    dict(
        type='CosineAnnealingLR',
        T_max=6,
        eta_min=lr * 1e-4,
        begin=4,
        end=10,
        by_epoch=True,
        convert_to_iter_based=True),
    # momentum scheduler
    # During the first 8 epochs, momentum increases from 0 to 0.85 / 0.95
    # during the next 12 epochs, momentum increases from 0.85 / 0.95 to 1
    dict(
        type='CosineAnnealingMomentum',
        T_max=4,
        eta_min=0.85 / 0.95,
        begin=0,
        end=4,
        by_epoch=True,
        convert_to_iter_based=True),
    dict(
        type='CosineAnnealingMomentum',
        T_max=6,
        eta_min=1,
        begin=4,
        end=10,
        by_epoch=True,
        convert_to_iter_based=True)
]

# runtime settings, 8/29에 시험용으로 4/4회만 돌리는 용도로 바꿔놓았음!
## 정확한 시험 해보고싶으면 20/4로 변경해서 써라
train_cfg = dict(by_epoch=True, max_epochs=10, val_interval=5)
val_cfg = dict()
test_cfg = dict()

# Default setting for scaling LR automatically
#   - `enable` means enable scaling LR automatically
#       or not by default.
#   - `base_batch_size` = (8 GPUs) x (4 samples per GPU).
auto_scale_lr = dict(enable=False, base_batch_size=32)
