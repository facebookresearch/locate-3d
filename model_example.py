from models.locate_3d import Locate3D
from omegaconf import OmegaConf
from locate3d_data.locate3d_dataset import Locate3DDataset

# First run to build cache
# python -m preprocessing.run_preprocessing --l3dd_annotations_fpath locate3d_data/dataset/train_scannetpp.json --scannet_data_dir /fsx-cortex/shared/datasets/scannet_ac --scannetpp_data_dir /datasets01/scannetpp/07252024 --end 5

cfg = OmegaConf.load("config/locate_3d.yaml")

model = Locate3D(cfg)
breakpoint()
# Locate 3d model
# model.load_from_checkpoint("/fsx-cortex/sergioarnaud/accel-cortex-m2f-sergioarnaud/2025-01-23_07-03-42_lam3d_train_jan20-locatex/0/checkpoints/ema_epoch_34.pt")

# Locate 3D+ model
model.load_from_checkpoint("/fsx-cortex-datacache/shared/locate-3d-weights/locate-3d-plus.pt")

dataset = Locate3DDataset(
    annotations_fpath = 'locate3d_data/dataset/val_scannetpp.json',
    return_featurized_pointcloud = True,
    scannet_data_dir = '/fsx-cortex/shared/datasets/scannet_ac',
    scannetpp_data_dir = '/datasets01/scannetpp/07252024',
    arkitscenes_data_dir = '/datasets01/ARKitScenes/',
)

import pickle
import torch
scene1 = pickle.load(open("/fsx-cortex-datacache/adamartin/for_paul_apr_16/predictions1.pkl", "rb"))

olddata = torch.load(f"/fsx-cortex-datacache/shared/cache/cached_pointclouds/concat_samh_clip_dino_5cm_frameskip_30_pth_valfix/concept_fusion_clip_val_ScannetPPDataset/{scene1[1]}.pt")
olddata_formatted = {
    'points': olddata['points'],
    'rgb': olddata['points'],
    'features_clip': olddata['features'][:, :768],
    'features_dino': olddata['features'][:, 768:],
}

model.encoder(olddata_formatted)
breakpoint()
data = dataset[219]

output = model.inference(
    data['featurized_sensor_pointcloud'],
    data['lang_data']['text_caption']
)

breakpoint()

import pickle
import torch
scene1 = pickle.load(open("/fsx-cortex-datacache/adamartin/for_paul_apr_16/predictions1.pkl", "rb"))
scene2 = pickle.load(open("/fsx-cortex-datacache/adamartin/for_paul_apr_16/predictions2.pkl", "rb"))
scene3 = pickle.load(open("/fsx-cortex-datacache/adamartin/for_paul_apr_16/predictions3.pkl", "rb"))

scene_list = sorted(dataset.list_scenes())

scene_list.index(('ScanNet++', '5f99900f09', tuple(range(30*(11)*7, 30*(11)*8, 30) ))) # index 501


# python -m preprocessing.run_preprocessing --l3dd_annotations_fpath locate3d_data/dataset/val_scannetpp.json --scannet_data_dir /fsx-cortex/shared/datasets/scannet_ac --scannetpp_data_dir /datasets01/scannetpp/07252024 --start 501 --end 502

data = dataset[1587]


olddata = torch.load(f"/fsx-cortex-datacache/shared/cache/cached_pointclouds/concat_samh_clip_dino_5cm_frameskip_30_pth_valfix/concept_fusion_clip_val_ScannetPPDataset/{scene1[1]}.pt")

dataset2 = Locate3DDataset(
    annotations_fpath = 'locate3d_data/dataset/val_arkitscenes.json',
    return_featurized_pointcloud = True,
    scannet_data_dir = '/fsx-cortex/shared/datasets/scannet_ac',
    scannetpp_data_dir = '/datasets01/scannetpp/07252024',
    arkitscenes_data_dir = '/datasets01/ARKitScenes/',
)

scene_list2 = sorted(dataset2.list_scenes())

scene_list2.index(('ARKitScenes', '44358451', None)) # index 58


dataset3 = Locate3DDataset(
    annotations_fpath = 'locate3d_data/dataset/val_scannet.json',
    return_featurized_pointcloud = True,
    scannet_data_dir = '/fsx-cortex/shared/datasets/scannet_ac',
    scannetpp_data_dir = '/datasets01/scannetpp/07252024',
    arkitscenes_data_dir = '/datasets01/ARKitScenes/',
)

scene_list3 = sorted(dataset3.list_scenes())

scene_list3.index(('ScanNet', 'scene0599_00', None)) # index 92


# python -m preprocessing.run_preprocessing --l3dd_annotations_fpath locate3d_data/dataset/val_scannet.json --scannet_data_dir /fsx-cortex/shared/datasets/scannet_ac --scannetpp_data_dir /datasets01/scannetpp/07252024 --start 92 --end 93

# python -m preprocessing.run_preprocessing --l3dd_annotations_fpath locate3d_data/dataset/val_arkitscenes.json --scannet_data_dir /fsx-cortex/shared/datasets/scannet_ac --scannetpp_data_dir /datasets01/scannetpp/07252024 --start 58 --end 59 --arkitscenes_data_dir /datasets01/ARKitScenes

# [i['description'] for i in dataset3.annos].index(scene3[0]) # 0

data3 = dataset3[0]

# [i['description'] for i in dataset2.annos].index(scene2[0]) # 277

data2 = dataset2[277]

output2 = model.inference(
    data2['featurized_sensor_pointcloud'],
    data2['lang_data']['text_caption']
)
output3 = model.inference(
    data3['featurized_sensor_pointcloud'],
    data['lang_data']['text_caption']
)

data2['featurized_sensor_pointcloud'] = {key: value.float() for key, value in data2['featurized_sensor_pointcloud'].items()}


scene1nd = pickle.load(open("/fsx-cortex-datacache/adamartin/for_paul_apr_16/predictions_nodropout1.pkl", "rb"))
scene2nd = pickle.load(open("/fsx-cortex-datacache/adamartin/for_paul_apr_16/predictions_nodropout2.pkl", "rb"))
scene3nd = pickle.load(open("/fsx-cortex-datacache/adamartin/for_paul_apr_16/predictions_nodropout3.pkl", "rb"))


olddata_formatted = {
    'points': olddata['points'],
    'rgb': olddata['points'],
    'features_clip': olddata['features'][:, :768],
    'features_dino': olddata['features'][:, 768:],
}

output = model.inference(
    olddata_formatted,
    data['lang_data']['text_caption']
)
# >>> output[0]['bbox']
# tensor([0.2416, 3.0832, 0.0430, 1.9382, 5.2575, 2.4240], device='cuda:0')
# >>> output[0]['bbox']
# tensor([0.6302, 2.8028, 0.0385, 2.2254, 4.7831, 2.2712], device='cuda:0')
# >>> output[0]['bbox']
# tensor([0.2331, 3.5342, 0.0988, 1.7610, 5.3182, 2.1695], device='cuda:0')



obs1 = pickle.load(open('/fsx-cortex-datacache/adamartin/for_paul_apr_16/obs_snpp.pkl', "rb"))

j3d = model.encoder(olddata_formatted)


state_dict = torch.load("/fsx-cortex/adamartin/accel-cortex-m2f-adamartin/2025-01-24_20-05-14_lam3d_train_jan20-locatex_plus/0/checkpoints/ema_epoch_32.pt")['model_state_dict']
cleaned_state_dict = {}
for k, v in state_dict.items():
    if k.startswith("module."):
        cleaned_state_dict[k[7:]] = v  # Remove first 7 characters ('module.')
    else:
        cleaned_state_dict[k] = v

state_dict = cleaned_state_dict        
cleaned_state_dict = {}
for k, v in state_dict.items():
    k = k.replace("encoder.model.", "encoder.")
    cleaned_state_dict[k] = v


torch.save({"model_state_dict": cleaned_state_dict}, "/fsx-cortex-datacache/shared/locate-3d-weights/locate-3d-plus.pt")

#model.load_from_checkpoint("/fsx-cortex/sergioarnaud/accel-cortex-m2f-sergioarnaud/2025-01-23_07-03-42_lam3d_train_jan20-locatex/0/checkpoints/ema_epoch_34.pt")
state_dict = torch.load("/fsx-cortex/sergioarnaud/accel-cortex-m2f-sergioarnaud/2025-01-23_07-03-42_lam3d_train_jan20-locatex/0/checkpoints/ema_epoch_34.pt")['model_state_dict']
cleaned_state_dict = {}
for k, v in state_dict.items():
    if k.startswith("module."):
        cleaned_state_dict[k[7:]] = v  # Remove first 7 characters ('module.')
    else:
        cleaned_state_dict[k] = v

state_dict = cleaned_state_dict        
cleaned_state_dict = {}
for k, v in state_dict.items():
    k = k.replace("encoder.model.", "encoder.")
    cleaned_state_dict[k] = v


torch.save({"model_state_dict": cleaned_state_dict}, "/fsx-cortex-datacache/shared/locate-3d-weights/locate-3d.pt")



        
jepa_sd = {}
for k, v in cleaned_state_dict.items():
    if "encoder." == k[:8]:
        jepa_sd[k[8:]] = v


torch.save({"target_encoder": jepa_sd}, "/fsx-cortex-datacache/shared/locate-3d-weights/encoder_3djepa_finetuned.pt")

e3djepa = torch.load("/fsx-cortex/pmcvay/jepa/fixintrinsics-clipdino-ptv3-sparsepred-percent8-sn-arkit/jepa-e480.pth.tar")

torch.save({"target_encoder": e3djepa['target_encoder']}, "/fsx-cortex-datacache/shared/locate-3d-weights/encoder_3djepa_pretrained.pt")
