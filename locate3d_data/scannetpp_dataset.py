import torch
import pyminiply
import json

class ScanNetPPDataset:
    def __init__(self, dataset_path):
        self.dataset_path = dataset_path

    def get_scan(self, scene_name):
        # Load corresponding PLY
        ppath = f'{self.dataset_path}/data/{scene_name}/scans/mesh_aligned_0.05.ply'
        xyz, _, _, _, rgb = pyminiply.read(ppath)
        xyz = torch.tensor(xyz)
        rgb = torch.tensor(rgb)

        # Load corresponding object segments anno
        with open(f"{self.dataset_path}/data/{scene_name}/scans/segments_anno.json", 'r') as f:
            a = json.load(f)
        seg_groups = a['segGroups']


        seg = torch.zeros_like(xyz[:, 0])
        seg -= 1

        for idx, sg in enumerate(seg_groups):
            assert sg['id'] == sg['objectId']
            seg[sg['segments']] = sg['id']
        return xyz, rgb, seg
