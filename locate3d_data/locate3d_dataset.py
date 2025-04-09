import json
from typing import Any, Dict

import torch

from locate3d_data.scannet_dataset import ScanNetDataset
from locate3d_data.scannetpp_dataset import ScanNetPPDataset
from locate3d_data.arkitscenes_dataset import ARKitScenesDataset

class Locate3DDataset:
    def __init__(
        self,
        annotations_fpath : str,
        scannet_align_matrix_path: str,
        scannet_data_dir: str,
        scannetpp_data_dir: str,
        arkitscenes_data_dir: str,
    ):
        super().__init__()
        
        self.scannet_dataset = None
        self.scannetpp_dataset = None
        self.arkitscenes_dataset = None

        if scannet_data_dir:
            self.scannet_dataset = ScanNetDataset(scannet_data_dir)
        if scannetpp_data_dir:
            self.scannetpp_dataset = ScanNetPPDataset(scannetpp_data_dir)
        if arkitscenes_data_dir:
            self.arkitscenes_dataset = ARKitScenesDataset(arkitscenes_data_dir)

        with open(annotations_fpath) as f:
            self.annos = json.load(f)

        if scannet_align_matrix_path is None:
            self.align_matrices = None
        else:
            with open(scannet_align_matrix_path, "r") as f:
                self.align_matrices = json.load(f)

    def _get_utterance_char_range(self, tokens, token_idxs):
        '''
        Convert from token indices to character indices in the utterance.
        '''
        first_token_idx = token_idxs[0]
        start_index = len(" ".join(tokens[:token_idxs[0]]))
        if first_token_idx > 0:
            start_index += 1  # plus one for space
        
        last_token_idx = token_idxs[-1]
        end_index = len(" ".join(tokens[:last_token_idx])) + len(
            tokens[last_token_idx]
        )
        if last_token_idx > 0:
            end_index += 1  # plus one because the span is exclusive

        return [start_index, end_index]

    def add_positive_map_and_obj_ids(self, dataset_dict):
        '''
        Add positive map and object IDs to the dataset dictionary.
        Processes the dataset dictionary to extract object IDs and their token ranges in the utterance.
        '''

        target_obj_id = int(dataset_dict["object_id"])
        tokens = dataset_dict["token"]
        utterance = " ".join(tokens)
        tokens_positive = []  # Character spans of tokens corresponding to each object ID
        object_ids = []  # Object ID (corresponds to ScanNet/ScanNet++ instance mask ID)

        for entity in dataset_dict["entities"]:
            _token_idxs, _entity_names = sorted(entity[0]), entity[1]
            utterance_range = self._get_utterance_char_range(tokens, _token_idxs)

            for entity_name in _entity_names:
                obj_id = int(entity_name.split("_")[0])
                is_target = obj_id == target_obj_id
                is_new_object = obj_id not in object_ids

                if is_new_object:
                    position = 0 if is_target else len(tokens_positive)
                    tokens_positive.insert(position, [utterance_range])
                    object_ids.insert(position, obj_id)
                else:
                    index = object_ids.index(obj_id)
                    tokens_positive[index].append(utterance_range)

        assert len(object_ids) == len(tokens_positive)

        dataset_dict["utterance"] = utterance
        dataset_dict["object_ids"] = object_ids
        dataset_dict["tokens_positive"] = tokens_positive

        return dataset_dict

    def align_ptc_to_camera(self, point_cloud, align_matrix):
        """
        Reorients pointcloud to align with RGBD+pose camera data using provided realignment matrix.
        alignment matrix @ [x, y, z, 1]^T = [x', y', z', 1]^T
        inputs:
            point_cloud: N X 3
            align_matrix: 4 X 4
        returns:
            point_cloud: N X 3
        """

        point_cloud = torch.cat(
            [point_cloud, torch.ones_like(point_cloud[..., :1])], dim=-1
        )
        return torch.matmul(point_cloud, align_matrix.T)[..., :3]

    def generate_scene_language_data(self, dataset_dict, scene_data):
        """
        Take in output of add_positive_map_and_obj_ids and combine with scene data to produce masks and boxes.
        """
        dataset_dict = self.add_positive_map_and_obj_ids(dataset_dict)
        tokens_positive = dataset_dict["tokens_positive"]

        utterance = dataset_dict["utterance"].lower()
        all_ids = dataset_dict["object_ids"]

        # extract the relevant masks from the scene data

        # There is one sample in ScanEnts-ScanRefer which has no entities.
        if len(all_ids) > 0:
            if 'seg' not in scene_data:
                masks = None
                boxes = torch.tensor([dataset_dict['gt_boxes'][_id] for _id in all_ids])
            else:
                masks = torch.stack([scene_data['seg'] == _id for _id in all_ids])

                n_instances = len(all_ids)
                boxes = torch.empty(size=(n_instances, 3, 2)) - torch.inf # Boxes associated with empty masks are all -inf.
                for i in range(n_instances):
                    masked_points = scene_data['xyz'][masks[i] > 0]
                    boxes[i, :, 0] = masked_points.min(axis=0)[0]
                    boxes[i, :, 1] = masked_points.max(axis=0)[0]

        else:
            masks = None
            boxes = None

        return {
            "text_caption": utterance,
            "positive_map": tokens_positive,
            "gt_masks": masks,
            "gt_boxes": boxes,
        }

    def load_scannet_scene_data(self, scene_name):
        assert self.scannet_dataset is not None, "ScanNet dataset not loaded."
        xyz, rgb, seg, _ = self.scannet_dataset.get_scan(scene_name)

        if self.align_matrices is not None:
            align_matrix = torch.tensor(self.align_matrices[scene_name]).reshape(4, 4)
            xyz = self.align_ptc_to_camera(xyz, align_matrix)

        return {"xyz": xyz, "rgb": rgb, "seg": seg}

    def load_scannetpp_scene_data(self, scene_name):
        assert self.scannetpp_dataset is not None, "ScanNet++ dataset not loaded."
        xyz, rgb, seg = self.scannetpp_dataset.get_scan(scene_name)
        return {"xyz": xyz, "rgb": rgb, "seg": seg}

    def load_arkitscenes_scene_data(self, scene_name):
        assert self.arkitscenes_dataset is not None, "ARKitScenes dataset not loaded."
        xyz, rgb = self.arkitscenes_dataset.get_scan(scene_name)
        return {"xyz": xyz, "rgb": rgb}


    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """
        Load an annotation and the corresponding scene data.
        """
        # Get the scene name
        anno = self.annos[idx]
        scene_name = anno["scene_id"]
        if "scene_dataset" in anno:
            scene_dataset = anno['scene_dataset']
            assert scene_dataset in ['ScanNet', 'ScanNet++', 'ARKitScenes'], "Unknown scene dataset"
        else:
            scene_dataset = 'ScanNet' # For compatibility with ScanEnts-ScanRefer
        if scene_dataset == 'ScanNet':
            scene_data = self.load_scannet_scene_data(scene_name)
        elif scene_dataset == 'ScanNet++':
            scene_data = self.load_scannetpp_scene_data(scene_name)
        elif scene_dataset == 'ARKitScenes':
            scene_data = self.load_arkitscenes_scene_data(scene_name)
        lang_data = self.generate_scene_language_data(anno, scene_data)

        return {
                "scene_name": scene_name,
                **scene_data,
                **lang_data,
            }

    def __len__(self):
        """Return number of utterances."""
        return len(self.annos)
