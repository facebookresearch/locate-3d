import torch

from models.ptv3_wrapper import PTv3Wrapper

model_path = '/fsx-cortex/pmcvay/jepa/fixintrinsics-clipdino-ptv3-sparsepred-percent8-sn-arkit/jepa-e480.pth.tar'

model = PTv3Wrapper(
    input_feat_dim = 1536,
    embed_dim = 256,
    rgb_proj_dim = 256,
    num_rgb_harmonic_functions=16,
    ptv3_args = {
        'dec_channels': [256, 256, 512, 512],
        'enc_channels': [256, 256, 512, 512, 1024],
        'in_channels': 256,
    }
)

model.load_weights(model_path)
model = model.cuda()

# Run model TODO: change to new data annotations we are releasing
import sys
sys.path.append("..")
from cortex.datasets.cached_dataset import CachedDataset

dataset = CachedDataset('concept_fusion_clip_val_ScanNetDataset', cache_path = '/fsx-cortex-datacache/shared/cache/cached_pointclouds/concat_samh_clip_dino_5cm_frameskip_30_pth_valfix', skip_fields = ["bboxes_3d", "frame_history", "objects", "latents"], use_cpp = True, use_scratch_cache=False)

data = {
    'points': dataset[0].observations.pointcloud.points_reduced.unsqueeze(0).cuda(),
    'features': dataset[0].observations.pointcloud.features_reduced.unsqueeze(0).cuda(),
    'rgb': dataset[0].observations.pointcloud.rgb_reduced.unsqueeze(0).cuda(),
}

# open source output

open_source_output = model(data)

# Run old version to verify
from omegaconf import OmegaConf
from cortex.observation_encoders.cortex_model_transform import CortexModelTransform

cfg = OmegaConf.create({'weights_path': model_path, 'load_weights': True})

model_old = CortexModelTransform(cfg)

old_output = model_old.transform_observations(dataset[0].observations)

# Compare outputs
torch.max((open_source_output - old_output[0].pointcloud.features_reduced) ** 2)

