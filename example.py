from preprocessing.datasets.scannet_dataset import ScanNetDataset
from preprocessing.datasets.arkitscenes_dataset import ARKitScenesDataset
from locate3d_data.scannet_dataset import ScanNetDataset as ScanNetDatasetL3DD
from locate3d_data.arkitscenes_dataset import ARKitScenesDataset as ARKitScenesDatasetL3DD
from locate3d_data.locate3d_dataset import Locate3DDataset

dataset = ARKitScenesDataset(root_dir = '/datasets01/ARKitScenes/raw/', frame_skip=30)

d1 = dataset[0]

l3d_dataset = ARKitScenesDatasetL3DD('/datasets01/ARKitScenes/')

d2 = l3d_dataset.get_camera_views('40753679')
breakpoint()

dataset = ScanNetDataset(root_dir = '/fsx-cortex/shared/datasets/scannet_ac', split = 'val', frame_skip = 30, n_classes = 549)

d1 = dataset[0]

l3ddscannet = ScanNetDatasetL3DD('/fsx-cortex/shared/datasets/scannet_ac')

d2 = l3ddscannet.get_camera_views('scene0011_00')

# TODO: correct except for the sky rotations

l3ddd = Locate3DDataset(annotations_fpath = 'locate3d_data/dataset/train_scannet.json', scannet_data_dir = '/fsx-cortex/shared/datasets/scannet_ac')
a = l3ddd.list_scenes()

cv = l3ddd.get_camera_views(*a[0])

# TODO: how to chunk?

from preprocessing.pointcloud_featurizer import FeatureLifter3D
from omegaconf import OmegaConf

pointcloud_featurizer_dino_cfg = OmegaConf.load("preprocessing/config/dino.yaml")
pointcloud_featurizer_dino = FeatureLifter3D(pointcloud_featurizer_dino_cfg)

ptc = pointcloud_featurizer_dino.lift_frames(cv)

breakpoint()
pointcloud_featurizer_clip_cfg = OmegaConf.load("preprocessing/config/clip.yaml")
pointcloud_featurizer_clip = FeatureLifter3D(pointcloud_featurizer_clip_cfg)

# TODO: sam weigths path and download instructions
ptc = pointcloud_featurizer_clip.lift_frames(cv)

# TODO: do we want frame cache?
breakpoint()
