# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the CC-by-NC license found in the
# LICENSE file in the root directory of this source tree.

import os
import sys
import torch
import traceback
from locate3d_data.locate3d_dataset import Locate3DDataset
from models.encoder_3djepa import Encoder3DJEPA
from models.locate_3d import downsample

# Set paths to data directories (update these paths as needed)
SCANNET_DATA_DIR = "[scannet_data_dir]"  # Replace with actual path
SCANNETPP_DATA_DIR = "[scannetpp_data_dir]"  # Replace with actual path
ARKITSCENES_DATA_DIR = "[arkitscenes_data_dir]"  # Replace with actual path

# Downsampling parameters
MAX_POINTS = 20000  # Reduced from 30000 to avoid OOM

# Load ScanNet dataset
dataset = Locate3DDataset(
    annotations_fpath="locate3d_data/dataset/train_scannet.json",  # or val_scannet.json
    return_featurized_pointcloud=True,
    scannet_data_dir=SCANNET_DATA_DIR,
    scannetpp_data_dir=SCANNETPP_DATA_DIR,
    arkitscenes_data_dir=ARKITSCENES_DATA_DIR,
)

# Load 3D JEPA model
model_3djepa = Encoder3DJEPA.from_pretrained("facebook/3d-jepa")
model_3djepa.eval()  # Set to evaluation mode
if torch.cuda.is_available():
    device = torch.device("cuda")
    model_3djepa = model_3djepa.to(device)  # Move to GPU
else:
    device = torch.device("cpu")
    print("CUDA not available, using CPU")

# Create output directory
output_dir = "outputs/3d_jepa_embeddings"
os.makedirs(output_dir, exist_ok=True)

# Build the list of unique-scene annotation indices up front so we don't pay
# the cost of loading a scene (mesh + featurized pointcloud from disk) for
# every duplicate annotation. Dedup key matches the cache-file key used by
# Locate3DDataset: (scene_dataset, scene_id, frames_used).
seen_scene_keys = set()
unique_indices = []
for idx, anno in enumerate(dataset.annos):
    scene_dataset = Locate3DDataset.get_scene_dataset_from_annotation(anno)
    frames_used = tuple(anno["frames_used"]) if "frames_used" in anno else None
    key = (scene_dataset, anno["scene_id"], frames_used)
    if key in seen_scene_keys:
        continue
    seen_scene_keys.add(key)
    unique_indices.append(idx)

print(f"Total annotations: {len(dataset)}, unique scenes: {len(unique_indices)}")

failed_scenes = []

for idx in unique_indices:
    try:
        data = dataset[idx]
        scene_name = data["scene_name"]

        # Ensure data is on the correct device
        if isinstance(data["featurized_sensor_pointcloud"], dict):
            data["featurized_sensor_pointcloud"] = {k: v.to(device) if hasattr(v, 'to') else v for k, v in data["featurized_sensor_pointcloud"].items()}
        elif hasattr(data["featurized_sensor_pointcloud"], 'to'):
            data["featurized_sensor_pointcloud"] = data["featurized_sensor_pointcloud"].to(device)

        # Align tensor lengths: CLIP/DINO featurization in preprocessing can
        # produce point counts that differ by a few, which breaks torch.cat
        # inside the encoder. Only the reduced tensors that the encoder
        # actually consumes need to share a length; *_original keys are the
        # pre-reduction pointcloud and must be left alone.
        pc_dict = data["featurized_sensor_pointcloud"]
        if isinstance(pc_dict, dict):
            reduced_keys = ("points", "rgb", "features_clip", "features_dino")
            lengths = {
                k: pc_dict[k].shape[0]
                for k in reduced_keys
                if k in pc_dict and hasattr(pc_dict[k], "shape") and pc_dict[k].ndim > 0
            }
            if lengths and min(lengths.values()) != max(lengths.values()):
                min_len = min(lengths.values())
                print(f"  ! Aligning mismatched reduced-tensor lengths for {scene_name}: {lengths} -> {min_len}")
                pc_dict = {
                    k: (v[:min_len] if k in lengths else v)
                    for k, v in pc_dict.items()
                }
                data["featurized_sensor_pointcloud"] = pc_dict

        # Downsample pointcloud
        featurized_pc = downsample(data["featurized_sensor_pointcloud"], MAX_POINTS)
        
        try:
            # Check input data
            if isinstance(featurized_pc, dict):
                # Log input info for debugging
                print(f"\nProcessing scene {scene_name}:")
                print(f"  Dict keys: {list(featurized_pc.keys())}")
                if "points" in featurized_pc:
                    print(f"  Points shape: {featurized_pc['points'].shape}")
                    print(f"  Points device: {featurized_pc['points'].device}")
                if "features_clip" in featurized_pc:
                    print(f"  Features_clip shape: {featurized_pc['features_clip'].shape}")
                    print(f"  Features_clip device: {featurized_pc['features_clip'].device}")
            else:
                print(f"  Input type: {type(featurized_pc)}")
            
            # Run encoder with no_grad
            with torch.no_grad():
                model_output = model_3djepa(featurized_pc)
            
            output = {
                "features": model_output.cpu() if hasattr(model_output, 'cpu') else model_output,
                "points": featurized_pc["points"].cpu() if hasattr(featurized_pc["points"], 'cpu') else featurized_pc["points"]
            }
            
            # Save output
            output_path = os.path.join(output_dir, f"{scene_name}.pt")
            torch.save(output, output_path)
            print(f"  ✓ Saved embedding to {output_path}")
            
        except RuntimeError as model_e:
            print(f"  ✗ CUDA/Runtime error for scene {scene_name}:")
            print(f"    {str(model_e)}")
            failed_scenes.append((scene_name, "CUDA/Runtime", str(model_e)))
            continue
        except Exception as model_e:
            print(f"  ✗ Model error for scene {scene_name}:")
            print(f"    {type(model_e).__name__}: {str(model_e)}")
            failed_scenes.append((scene_name, type(model_e).__name__, str(model_e)))
            traceback.print_exc()
            continue
        
    except FileNotFoundError as data_e:
        print(f"Skipped scene at index {idx} (missing cache file): {data_e}")
        continue
    except Exception as data_e:
        print(f"Skipped scene at index {idx}: {type(data_e).__name__}: {data_e}")
        continue

print("\n" + "="*60)
print("Processing complete.")
print(f"Total unique scenes attempted: {len(unique_indices)}")
print(f"Total scenes succeeded: {len(unique_indices) - len(failed_scenes)}")
if failed_scenes:
    print(f"Failed scenes: {len(failed_scenes)}")
    for scene, error_type, error_msg in failed_scenes:
        print(f"  - {scene}: {error_type}")
print("="*60)
