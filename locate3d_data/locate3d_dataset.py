# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the CC-by-NC license found in the
# LICENSE file in the root directory of this source tree.


import json
from typing import Any, Dict

import torch
import os

from locate3d_data.scannet_dataset import ScanNetDataset
from locate3d_data.scannetpp_dataset import ScanNetPPDataset
from locate3d_data.arkitscenes_dataset import ARKitScenesDataset


class Locate3DDataset:
    def __init__(
        self,
        annotations_fpath: str,
        return_featurized_pointcloud: bool,
        scannet_data_dir: str = None,
        scannetpp_data_dir: str = None,
        arkitscenes_data_dir: str = None,
        cache_path: str = "cache",
    ):
        super().__init__()

        self.scannet_dataset = None
        self.scannetpp_dataset = None
        self.arkitscenes_dataset = None
        self.return_featurized_pointcloud = return_featurized_pointcloud
        self.cache_path = cache_path

        if scannet_data_dir:
            self.scannet_dataset = ScanNetDataset(scannet_data_dir)
        if scannetpp_data_dir:
            self.scannetpp_dataset = ScanNetPPDataset(scannetpp_data_dir)
        if arkitscenes_data_dir:
            self.arkitscenes_dataset = ARKitScenesDataset(arkitscenes_data_dir)

        with open(annotations_fpath) as f:
            self.annos = json.load(f)

    def _get_utterance_char_range(self, tokens, token_idxs):
        """
        Convert from token indices to character indices in the utterance.
        """
        first_token_idx = token_idxs[0]
        start_index = len(" ".join(tokens[: token_idxs[0]]))
        if first_token_idx > 0:
            start_index += 1  # plus one for space

        last_token_idx = token_idxs[-1]
        end_index = len(" ".join(tokens[:last_token_idx])) + len(tokens[last_token_idx])
        if last_token_idx > 0:
            end_index += 1  # plus one because the span is exclusive

        return [start_index, end_index]

    def add_positive_map_and_obj_ids(self, dataset_dict):
        """
        Add positive map and object IDs to the dataset dictionary.
        Processes the dataset dictionary to extract object IDs and their token ranges in the utterance.
        """

        target_obj_id = int(dataset_dict["object_id"])
        tokens = dataset_dict["token"]
        utterance = " ".join(tokens)
        tokens_positive = (
            []
        )  # Character spans of tokens corresponding to each object ID
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
            if "seg" not in scene_data:
                masks = None
                boxes = torch.tensor([dataset_dict["gt_boxes"][_id] for _id in all_ids])
            else:
                masks = torch.stack([scene_data["seg"] == _id for _id in all_ids])

                n_instances = len(all_ids)
                boxes = (
                    torch.empty(size=(n_instances, 3, 2)) - torch.inf
                )  # Boxes associated with empty masks are all -inf.
                for i in range(n_instances):
                    masked_points = scene_data["xyz"][masks[i] > 0]
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
        return {"xyz": xyz, "rgb": rgb, "seg": seg}

    def load_scannetpp_scene_data(self, scene_name):
        assert self.scannetpp_dataset is not None, "ScanNet++ dataset not loaded."
        xyz, rgb, seg = self.scannetpp_dataset.get_scan(scene_name)
        return {"xyz": xyz, "rgb": rgb, "seg": seg}

    def load_arkitscenes_scene_data(self, scene_name):
        assert self.arkitscenes_dataset is not None, "ARKitScenes dataset not loaded."
        xyz, rgb = self.arkitscenes_dataset.get_scan(scene_name)
        return {"xyz": xyz, "rgb": rgb}

    @staticmethod
    def get_scene_dataset_from_annotation(anno):
        if "scene_dataset" in anno:
            scene_dataset = anno["scene_dataset"]
            assert scene_dataset in [
                "ScanNet",
                "ScanNet++",
                "ARKitScenes",
            ], "Unknown scene dataset"
        else:
            scene_dataset = "ScanNet"  # For compatibility with ScanEnts-ScanRefer

        return scene_dataset

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """
        Load an annotation and the corresponding scene data.
        """
        # Get the scene name
        anno = self.annos[idx]
        scene_name = anno["scene_id"]
        scene_dataset = Locate3DDataset.get_scene_dataset_from_annotation(anno)

        if scene_dataset == "ScanNet":
            scene_data = self.load_scannet_scene_data(scene_name)
        elif scene_dataset == "ScanNet++":
            scene_data = self.load_scannetpp_scene_data(scene_name)
        elif scene_dataset == "ARKitScenes":
            scene_data = self.load_arkitscenes_scene_data(scene_name)
        lang_data = self.generate_scene_language_data(anno, scene_data)

        if not self.return_featurized_pointcloud:
            return {
                "scene_name": scene_name,
                "mesh": {**scene_data},
                **lang_data,
            }

        if "frames_used" in anno:
            frames_used = anno["frames_used"]
            cache_file = os.path.join(
                self.cache_path,
                scene_dataset,
                f"{scene_name}_start{frames_used[0]}_end{frames_used[-1]}.pt",
            )
        else:
            cache_file = os.path.join(
                self.cache_path, scene_dataset, f"{scene_name}.pt"
            )

        assert os.path.exists(
            cache_file
        ), "Must first run preprocessing to load featurized pointcloud"

        featurized_pointcloud = torch.load(cache_file)

        return {
            "scene_name": scene_name,
            "mesh": {**scene_data},
            "featurized_sensor_pointcloud": {**featurized_pointcloud},
            "lang_data": {**lang_data},
        }

    def __len__(self):
        """Return number of utterances."""
        return len(self.annos)

    def get_camera_views(self, scene_dataset, scene_name, frames_used):
        dataset = None
        if scene_dataset == "ScanNet":
            return self.scannet_dataset.get_camera_views(scene_name)
        elif scene_dataset == "ScanNet++":
            return self.scannetpp_dataset.get_camera_views(scene_name, frames_used)
        elif scene_dataset == "ARKitScenes":
            return self.arkitscenes_dataset.get_camera_views(scene_name)
        else:
            raise Exception("Specified dataset not supported")

    def list_scenes(self):
        scenes = set()

        for anno in self.annos:
            scene_name = anno["scene_id"]
            scene_dataset = Locate3DDataset.get_scene_dataset_from_annotation(anno)
            if "frames_used" in anno:
                key = (scene_dataset, scene_name, tuple(anno["frames_used"]))
            else:
                key = (scene_dataset, scene_name, None)
            scenes.add(key)

        return list(scenes)
