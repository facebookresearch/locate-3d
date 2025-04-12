from typing import Any, Dict, List, Optional, Tuple, Union
from natsort import natsorted
from pathlib import Path
import numpy as np
import torch
import pyminiply
import json

from locate3d_data.data_utils import get_image_from_path, get_depth_image_from_path

class ScanNetDataset:
    DEPTH_SCALE_FACTOR = 0.001  # to MM
    DEFAULT_HEIGHT = 968.0
    DEFAULT_WIDTH = 1296.0

    def __init__(self, path, use_hi_quality_mesh=False):
        self.dataset_path = Path(path)
        self.use_hi_quality_mesh = use_hi_quality_mesh

        # Locate 3D default value
        self.frame_skip = 30 
        self.width = 640
        self.height = 480
        
        # Sub directories
        self.posed_dir = self.dataset_path / "posed_images"
        self.instance_dir = self.dataset_path / "scannet_instance_data"

    def get_scan(self, scene_id):
        scan_folder = self.dataset_path + "/scans"

        # Most of this logic is to deal with the fact that the .ply file does not directly contain the semantic segment as an argument
        # So we instead have to load the superpoints (non-semantic oversegmentation of scene), then load the map from superpoint to semantic segment

        prefix = f"{scan_folder}/{scene_id}/{scene_id}"
        if self.use_hi_quality_mesh:
            ply_file = f"{prefix}_vh_clean.ply"
            superpoint_json = f"{prefix}_vh_clean.segs.json"
            segments_json = f"{prefix}_vh_clean.aggregation.json"
        else:
            ply_file = f"{prefix}_vh_clean_2.ply"
            superpoint_json = f"{prefix}_vh_clean_2.0.010000.segs.json"
            segments_json = f"{prefix}.aggregation.json"
        xyz, _, _, _, rgb = pyminiply.read(ply_file)
        xyz = torch.tensor(xyz)
        rgb = torch.tensor(rgb)
        with open(superpoint_json) as f:
            superpoints = json.load(f)
        with open(segments_json) as f:
            segments = json.load(f)

        superpoints = torch.tensor(superpoints['segIndices'])

        seg = torch.zeros_like(superpoints) - 1

        id_to_label = dict() # Can probably be deleted later
        for group in segments['segGroups']:
            assert group['id'] == group['objectId']
            id_to_label[group['id']] = group['label'] # Can probably be deleted later
            for superpoint_idx in group['segments']:
                seg[superpoints == superpoint_idx] = group['id']

        assert len(xyz) == len(rgb) == len(seg)
        return xyz, rgb, seg, id_to_label

    def get_poses(
        self, scan_name, axis_align_mat
    ) -> List[torch.Tensor]:
        
        scene_pose_dir = self.posed_dir / scan_name
        scene_posed_files = [str(s) for s in scene_pose_dir.iterdir()]
        
        # Pose
        pose_names = list(
            natsorted(
                [
                    s
                    for s in scene_posed_files
                    if s.endswith(".txt") and not s.endswith("intrinsic.txt")
                ]
            )
        )[:: self.frame_skip]

        poses = []
        for p in pose_names:
            pose = np.loadtxt(p)
            pose = np.array(pose).reshape(4, 4)

            pose = axis_align_mat @ torch.from_numpy(pose.astype(np.float32)).float()
            
            if torch.any(torch.isnan(pose)):
                continue

            poses.append(pose)
            
        return torch.stack(poses).float()
    
    def get_intrinsics(
            self, scan_name, num_frames
    ) -> List[torch.Tensor]:
        intrinsic_name = self.posed_dir / scan_name / "intrinsic.txt"
        
        # Intrinsics shared across images
        K = torch.from_numpy(np.loadtxt(intrinsic_name).astype(np.float32))
        K[0] *= float(self.width) / self.DEFAULT_WIDTH  # scale_x
        K[1] *= float(self.height) / self.DEFAULT_HEIGHT  # scale_y
        K = K[:3, :3]
        intrinsics = torch.repeat_interleave(
            K.unsqueeze(0), repeats=num_frames, dim=0
        ).float()
        return intrinsics

    def get_images(
        self, scan_name
    ) -> List[torch.Tensor]:
        scene_pose_dir = self.posed_dir / scan_name
        scene_posed_files = [str(s) for s in scene_pose_dir.iterdir()]

        def get_endswith(f_list, endswith):
            return list(natsorted([s for s in f_list if s.endswith(endswith)]))

        img_names = get_endswith(scene_posed_files, ".jpg")[:: self.frame_skip]
        images = []

        for i in img_names:
            img = get_image_from_path(
                i, height=self.height, width=self.width
            )
            images.append(img)

        return torch.stack(images)


    def get_depths(
        self, scan_name
    ):
        scene_pose_dir = self.posed_dir / scan_name
        scene_posed_files = [str(s) for s in scene_pose_dir.iterdir()]

        def get_endswith(f_list, endswith):
            return list(natsorted([s for s in f_list if s.endswith(endswith)]))

        depth_names = get_endswith(scene_posed_files, ".png")[:: self.frame_skip]
        depths = []

        for d in depth_names:
            depth = get_depth_image_from_path(
                Path(d),
                height=self.height,
                width=self.width,
                scale_factor=self.DEPTH_SCALE_FACTOR,
            )
            depths.append(depth)

        return torch.stack(depths).float()
    
    def get_camera_views(self, scan_name):
        axis_align_mat = torch.from_numpy(
            np.load(self.instance_dir / f"{scan_name}_axis_align_matrix.npy")
        ).float()

        return {
            "cam_to_world": self.get_poses(scan_name, axis_align_mat),
            "cam_K": self.get_intrinsics(scan_name, 80),
            "rgb" : self.get_images(scan_name),
            "depth_zbuffer": self.get_depths(scan_name),
        }
