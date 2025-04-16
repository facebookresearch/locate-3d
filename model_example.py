from models.locate_3d import Locate3D
from omegaconf import OmegaConf
from locate3d_data.locate3d_dataset import Locate3DDataset

# First run to build cache
# python -m preprocessing.run_preprocessing --l3dd_annotations_fpath locate3d_data/dataset/train_scannetpp.json --scannet_data_dir /fsx-cortex/shared/datasets/scannet_ac --scannetpp_data_dir /datasets01/scannetpp/07252024 --end 5

cfg = OmegaConf.load("config/locate_3d.yaml")

model = Locate3D(cfg)

model.load_from_checkpoint("/fsx-cortex/sergioarnaud/accel-cortex-m2f-sergioarnaud/2025-01-23_07-03-42_lam3d_train_jan20-locatex/0/checkpoints/ema_epoch_34.pt")

dataset = Locate3DDataset(
    annotations_fpath = 'locate3d_data/dataset/train_scannetpp.json',
    return_featurized_pointcloud = True,
    scannet_data_dir = '/fsx-cortex/shared/datasets/scannet_ac',
    scannetpp_data_dir = '/datasets01/scannetpp/07252024',
    arkitscenes_data_dir = '/datasets01/ARKitScenes/',
)

data = dataset[219]

output = model.inference(
    data['featurized_sensor_pointcloud'],
    data['lang_data']['text_caption']
)

breakpoint()

