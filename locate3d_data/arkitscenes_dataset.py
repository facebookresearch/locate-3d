import torch
import pyminiply
import os

class ARKitScenesDataset:
    def __init__(self, dataset_path):
        self.dataset_path = dataset_path

    def get_scan(self, scene_id):
        # Construct possible paths to the PLY file
        training_path = os.path.join(self.dataset_path, "raw", "Training", scene_id, f"{scene_id}_3dod_mesh.ply")
        validation_path = os.path.join(self.dataset_path, "raw", "Validation", scene_id, f"{scene_id}_3dod_mesh.ply")
        
        # Check if the file exists in Training or Validation
        if os.path.exists(training_path):
            ply_path = training_path
        elif os.path.exists(validation_path):
            ply_path = validation_path
        else:
            raise FileNotFoundError(f"PLY file for scene {scene_id} in ARKitScenes dataset not found.")
        xyz, _, _, _, rgb = pyminiply.read(ply_path)
        xyz = torch.tensor(xyz)
        rgb = torch.tensor(rgb)
        
        return xyz, rgb
