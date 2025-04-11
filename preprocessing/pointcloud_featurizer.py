import logging
import os
import re
from typing import Any, Dict, List, Optional

import hydra
import torch
from omegaconf import DictConfig, OmegaConf

from preprocessing.cached_dataset import CachedDataset
from preprocessing.datasets.base_transform import BaseTransform
from preprocessing.types import (
    Action,
    BoundingBoxes3D,
    CameraViewsData,
    Observations,
    PointCloudData,
    TrainingSample,
)
from preprocessing.voxelized_pointcloud import VoxelizedPointcloud

logger = logging.getLogger(__name__)
MAX_OBJ_NAME_LEN = 77


class BaseEncoder(torch.nn.Module, BaseTransform):
    def __init__(self, cfg: OmegaConf, overrides: Optional[Dict[str, Any]] = None):
        """
        Goes from a Observation to a latent tensor representation with shape (K, dim)
        """
        torch.nn.Module.__init__(self)
        BaseTransform.__init__(self, cfg, overrides)

    def forward(self, obs, goal=None):
        return self.transform_observations(obs, goal)[0]


class LearnedLocalizerEncoder(BaseEncoder):
    def _initialize_localizer(self):
        if "location_predictor" in self.cfg:
            self.location_predictor = hydra.utils.instantiate(
                self.cfg.location_predictor
            )

    def _localize_object_learned(
        self, object_name: str, ptc: PointCloudData, quantile=0.9
    ):
        """
        Takes as input an object name in string format and an Observations object.
        Computes the (1-quantile) most similar points according to the confidence of the learned
        location predictor.
        Returns the mean location of most similar points and the highest similarity location, as well
        as indicies, pointcloud and all similarity values.
        """
        ptc_locations = ptc.points_reduced
        logits, probs = self.location_predictor(
            [
                TrainingSample(
                    observations=Observations(pointcloud=ptc), goal=object_name
                )
            ]
        )
        similarity = probs[0].squeeze()

        k = round(similarity.numel() * (1 - quantile))
        values, indices = torch.topk(similarity, k, sorted=True)

        output = {
            "mean_location": torch.mean(ptc_locations[indices], 0),
            "highest_similarity_location": ptc_locations[indices[0]],
            "similarity": values,  # rough measure of confidence, will be low if no good match
            "pointcloud": ptc,
            "indices": indices,
        }
        return output

    def _point_to_bbox(self, point: torch.Tensor) -> torch.Tensor:
        offset = 0.01

        return torch.stack([point - offset, point + offset], dim=1)

    def _extract_bbox_text_descriptions(self, action: Action) -> List[str]:
        matches = re.findall(r"(?<=<bboxtext>).+?(?=</bboxtext>)", action.action)

        new_action = Action(
            action=re.sub(r"<bboxtext>.+?</bboxtext>", "[BB]", action.action)
        )

        return new_action, matches

    def _process_action(self, episode_id: int, action: Action) -> Action:
        if "<bboxtext>" not in action.action:
            return episode_id, action

        new_action, matches = self._extract_bbox_text_descriptions(action)

        bboxes = []
        for match in matches:
            if self.cfg.get("learned_location_predictor", True):
                output = self._localize_object_learned(
                    match, self.state_memory[episode_id]
                )
            else:  # Why is there a reference to Ponder?
                # TODO: check if zero_shot fn defined
                if "_localize_object_zero_shot" in dir(self):
                    output = self._localize_object_zero_shot(
                        match, self.state_memory[episode_id]
                    )
                else:
                    raise ValueError("zero-shot prediction with Ponder not supported")

            most_similar_point_bbox = self._point_to_bbox(
                output["highest_similarity_location"]
            )
            bboxes.append(most_similar_point_bbox)

        new_action.bounding_box = BoundingBoxes3D(bounds=torch.stack(bboxes))
        return episode_id, new_action


class FeatureLifter3DTransform(LearnedLocalizerEncoder):
    def _assert_observations(self, obs: Observations) -> None:
        """
        Run checks to make sure the agent can act on these observations
        """
        assert obs.frame_history is not None
        if obs.pointcloud is not None:
            logger.warning(
                "Observations contain a pointcloud, which is not expected. Overwriting."
            )

    def _transform_observations(
        self, obs: Observations, goal: Optional[str] = None
    ) -> (Observations, Optional[Dict]):
        """
        Runs the observation transformation. For example, for the sparse
        voxel map, take in full frame observations and return object images
        """
        _id = obs.frame_history.scene_id
        for _d in self.cached_datasets:
            if _d.exists(_id):
                new_obs = _d.get_scene(_id).observations
                if self.cfg.keep_frame_history and not self.use_cache_frame_history:
                    new_obs.frame_history = obs.frame_history
                return new_obs, new_obs.pointcloud
        logger.warning(f"Naive3DFeatureLifterTransform cache miss for {_id}")

        vpc = VoxelizedPointcloud(**self.voxelized_pointcloud_kwargs)

        N = obs.frame_history.rgb.shape[0] // self.batch_size + 1
        for i in range(N):
            start, end = i * self.batch_size, (i + 1) * self.batch_size
            self.feature_slam.add_batch_image(
                vpc,
                obs.frame_history.rgb[start:end],
                obs.frame_history.depth_zbuffer[start:end],
                obs.frame_history.cam_to_world[start:end],
                obs.frame_history.cam_K[start:end],
                frame_path=obs.frame_history.view_id[start:end],
            )
        (
            points_reduced,
            features_reduced,
            weights_reduced,
            rgb_reduced,
        ) = vpc.get_pointcloud()

        pointcloud = PointCloudData(
            points_reduced=points_reduced,
            features_reduced=features_reduced,
            weights_reduced=weights_reduced,
            rgb_reduced=rgb_reduced,
        )

        new_observation = Observations(pointcloud=pointcloud)
        if self.cfg.keep_frame_history:
            new_observation.frame_history = obs.frame_history
        else:
            new_observation.frame_history = CameraViewsData(
                scene_id=obs.frame_history.scene_id, view_id=[]
            )
        return new_observation, pointcloud

    def _localize_object_zero_shot(
        self, object_name: str, ptc: PointCloudData, quantile=0.9
    ):
        """
        Takes as input an object name in string format and an Observations object.
        Computes the (1-quantile) most similar points according to similarity between text embedding
        and pointcloud features.
        Returns the mean location of most similar points and the highest similarity location, as well
        as indicies, pointcloud and all similarity values.
        """
        ptc_features = ptc.features_reduced
        ptc_locations = ptc.points_reduced
        if len(object_name) > MAX_OBJ_NAME_LEN:
            logger.warn(
                f"Object name description was over {MAX_OBJ_NAME_LEN=} characters, the max context length of clip, and was shortened. {object_name=}"
            )

        textfeat = (
            self.feature_slam.image_feature_generator.image_text_encoder.encode_text(
                object_name[:MAX_OBJ_NAME_LEN]
            )
        )  # was hitting out of context error for text longer that 77 characters

        textfeat = torch.nn.functional.normalize(textfeat, dim=-1)
        textfeat = textfeat.unsqueeze(0)

        ptc_features = torch.nn.functional.normalize(ptc_features, dim=-1)
        textfeat = textfeat.to(ptc_features.device)
        similarity = torch.nn.CosineSimilarity(dim=-1)(textfeat, ptc_features).squeeze()
        similarity = (similarity - similarity.min()) / (
            similarity.max() - similarity.min() + 1e-12
        )

        k = round(similarity.numel() * (1 - quantile))
        values, indices = torch.topk(similarity, k, sorted=True)

        output = {
            "mean_location": torch.mean(ptc_locations[indices], 0),
            "highest_similarity_location": ptc_locations[indices[0]],
            "similarity": values,  # rough measure of confidence, will be low if no good match
            "pointcloud": ptc,
            "indices": indices,
        }
        return output

    def initialize(self):
        self._initialize_localizer()

        self.batch_size = self.cfg.get("unproject_frame_batch_size", 1)
        self.voxelized_pointcloud_kwargs = self.cfg.get("voxelized_pointcloud", {})

        self.use_cache_frame_history = self.cfg.get("use_cache_frame_history", False)
        self.do_not_use_cache = self.cfg.get("do_not_use_cache", False)
        self.cache_path = self.cfg.get("cache_path", None)

        cache_path = self.cfg.get("cache_path")
        if os.path.exists(cache_path) and not self.do_not_use_cache:
            default_cache_keys = os.listdir(cache_path)
            logger.warning(f"COULD load default cache path at {cache_path}")
        else:
            default_cache_keys = []
            logger.warning(f"COULD NOT LOAD DEFAULT CACHE PATH AT {cache_path}")

        self.cached_datasets = [
            CachedDataset(
                key=k,
                cache_path=cache_path,
            )
            for k in self.cfg.get("cache_keys", default_cache_keys)
        ]

        if isinstance(self.cfg.feature_slam, DictConfig):
            logger.info("instantiating feature slam", self.cfg.feature_slam)
            self.feature_slam = hydra.utils.instantiate(self.cfg.feature_slam)
            logger.info(
                f"Feature generator: {self.feature_slam.image_feature_generator}"
            )
        else:
            self.feature_slam = self.cfg.feature_slam
