## Self-Adaptive Voxelization 구현 

""" 
Self-Adaptive Voxelization이란? 
LIDAR로 인식한 기존의 3D Point Cloud Image에 대해, 
사물을 Voxel의 형태로 인식하는 Voxelization 방법을 조금 더 발전시킨 방법.  
고정된 Voxel을 생성하는 게 아닌, Point의 밀집 된 정도와 상대적 거리에 따라 Voxel을 동적으로 생성
"""


![image](https://github.com/user-attachments/assets/308f04de-24e5-43b1-ab16-ad59b119b497)

Self-Adaptive Voxelization(SAV)

![image](https://github.com/user-attachments/assets/086d8cda-23e0-4a92-9ddb-2364a9e17858)

Self-Adaptive Voxelization(SAV)을 Model에 적용한 경우

## Openmmlab-MMdetection3D 오픈소스 이용 

PointPillars, SECOND, PV_RCNN, MVX_Net, Part-A2
다음 Voxelization 딥러닝 모델들에 대해 테스트 진행 및 생성.

KiTTI Object Detection Benchmark Dataset – LiDAR Point Cloud
Object Class: Car, Pedestrian, Cyclist 

오픈소스 MMDetection3D에서 제공하는 기본 Config file & Model 사용
Training sample:3712, Test sample:3769, Training epoch:20
처음 Voxel size: [0.05, 0.05, 0.1], Pillar size: [0.16, 0.16, 4]


![image](https://github.com/user-attachments/assets/91283273-6c95-4350-9366-79f7e03f232e)
![image](https://github.com/user-attachments/assets/d247fbbe-b618-4fed-ac0b-b58d5d847489)


Point를 좀 더 세밀하게 인식할 수 있는 Adaptive-Voxelization


## File 적용
MMdetection3D 오픈소스 설치 후, 

File: mmdet3d - model - voxel encoder에 적용. 
Commit file 중 voxel encoder를 기입하여 사용, 필요시 변형 및 수정


