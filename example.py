from preprocessing.datasets.scannet_dataset import ScanNetDataset
from locate3d_data.scannet_dataset import ScanNetDataset as ScanNetDatasetL3DD

dataset = ScanNetDataset(root_dir = '/fsx-cortex/shared/datasets/scannet_ac', split = 'val', frame_skip = 30, n_classes = 549)

d1 = dataset[0]

l3dd = ScanNetDatasetL3DD('/fsx-cortex/shared/datasets/scannet_ac')

d2 = l3dd.get_camera_views('scene0011_00')

# TODO: correct except for the sky rotations


breakpoint()
