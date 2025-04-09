import torch
import pyminiply
import json

class ScanNetDataset:
    def __init__(self, path, use_hi_quality_mesh=False):
        self.dataset_path = path
        self.use_hi_quality_mesh = use_hi_quality_mesh

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