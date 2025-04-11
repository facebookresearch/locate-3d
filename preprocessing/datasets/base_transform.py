import logging
from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, Tuple

from omegaconf import OmegaConf

from preprocessing.types import Action, Observations, TrainingSample

logger = logging.getLogger(__name__)


class BaseTransform(ABC):
    def __init__(self, cfg: OmegaConf, overrides: Optional[Dict[str, Any]] = None):
        """
        Base init function. Child classes with implement the initialize
        method instead of overwriting init

        Args:
            cfg: Omegaconf config object
            overrides: Dict = any cfg values to overwrite

        cfg.store_data_for_process_action:

        This config value is needed for every transform.
        The idea behind this config value is that we have two potential
        use cases for using transforms.

        1. Using transforms in a dataset generation (i.e. a dataset for alignment
        training)
        2. Using transforms within the evaluation pipeline as a step between
        benchmmark and agent (i.e. benchmark -> transform -> agent)

        When the transform is used within dataset generation, processing actions
        is never needed and thus storing information for computing process action
        is not needed. However, when transformed are used between benchmark and
        agent the backwards transform, process_action, is needed.

        For example, when used in evaluation, the svm transform during the forward
        pass needs to store a mapping from instance_id to bbox. Then during the
        backwards pass, it uses the stored data to transform the text "goto(img_1)"
        to Action(text = "goto(BB1)", bbox = [x, y, z]).

        However, if SVM is used in dataset generation, the inverse action transform
        is never needed because the llm is trained to predict "goto(img_1)" and so
        it doesn't make sense to store any data in state_memory.

        This config value forces all transforms to set whether or not data is
        stored in a consistent manner so a each pipeline can overwrite the value
        depending on the pipeline executing.
        """

        self.cfg = cfg
        self.overrides = overrides if overrides is not None else {}

        self.set_overrides()
        if not hasattr(self.cfg, "store_data_for_process_action"):
            self.cfg.store_data_for_process_action = True
        if self.cfg.store_data_for_process_action:
            self.state_memory = {}

        self.initialize()

    def set_overrides(self):
        """
        Base method to override config values. Child classes can
        create their own version
        """

        for key, value in self.overrides.items():
            if key in self.cfg:
                self.cfg[key] = value
            else:
                logger.warning(f"Unknown key {key} in overrides in BaseTransform")

    def episode_step(
        self, episode_id: int, obs: Observations = None, goal: Optional[str] = None
    ) -> (int, Observations, Optional[str]):
        """
        Function to be called externally. Runs both the observation asserts
        and runs the internal _process_observations. Implemented this way to
        be a forcing function actually write observation asserts
        """
        if obs is None:
            return (episode_id, None, goal)

        new_obs, data_to_store = self.transform_observations(obs, goal)

        if self.cfg.store_data_for_process_action:
            assert (
                episode_id not in self.state_memory
            ), "behavior currently undefined for continuing episode with more observations"
            self.state_memory[episode_id] = data_to_store

        return (episode_id, new_obs, goal)

    @abstractmethod
    def _assert_observations(self, obs: Observations) -> None:
        """
        Run checks to make sure the agent can act on these observations
        """

    def _assert_sample(self, sample: TrainingSample) -> None:
        """
        Run checks to make sure the data sample can be transformed appropriately
        """
        self._assert_observations(sample.observations)

    @abstractmethod
    def _transform_observations(
        self, obs: Observations, goal: Optional[str] = None
    ) -> (Observations, Optional[Dict]):
        """
        Runs the observation transformation. For example, for the sparse
        voxel map, take in full frame observations and return object images

        Output:
        new_observations: the transformed observations
        data_to_store: Dict of data that will be needed for process actions
        """

    def _transform_sample(
        self, sample: TrainingSample
    ) -> (TrainingSample, Optional[Dict]):
        """
        Runs transformation on a sample. For example, can be used for dataset
        transforms that need accessing other field of sample (info, goal etc)

        Output:
        new_sample: the transformed sample
        data_to_store: Dict of data that will be needed for process actions
        """
        sample.observations, data_to_store = self._transform_observations(
            sample.observations, goal=sample.goal
        )
        return sample, data_to_store

    def transform_observations(
        self, obs: Observations, goal: Optional[str] = None
    ) -> (Observations, Optional[Dict]):
        """
        Child defined internal method to run the observation transformation.
        For example, for the sparse
        voxel map, take in full frame observations and return object images

        Output:
        new_observations: the transformed observations
        data_to_store: Dict of data that will be needed for process actions
        """
        self._assert_observations(obs)
        return self._transform_observations(obs, goal)

    def transform_sample(
        self, sample: TrainingSample
    ) -> (TrainingSample, Optional[Dict]):
        """
        Child defined internal method to run the sample transformation.

        Output:
        new_sample: the transformed sample
        data_to_store: Dict of data that will be needed for process actions
        """
        self._assert_sample(sample)
        return self._transform_sample(sample)

    def process_action(self, episode_id: int, action: Action) -> Tuple[int, Action]:
        if self.cfg.store_data_for_process_action:
            return self._process_action(episode_id, action)

        raise ValueError(
            "cfg.store_data_for_process_action is set to false. Cannot call process action"
        )

    @abstractmethod
    def _process_action(self, episode_id: int, action: Action) -> Tuple[int, Action]:
        """
        Method for transform to modify returning action. For example,
        converting text to bounding boxes

        For example:

        For SVM, the transform builds a dict of crops and
        bounding boxes. The agent could select the crop by outputting
        an action with just text. I.e. Action(action = "goto(crop_3)")

        The transform then needs to convert "goto(crop_3)" looking up
        crop_3's bounding box and return Action(action = "goto([BB])",
        bounding_box = crop_3_bbox)

        """

    def episode_complete(self, episode_id: int) -> None:
        if episode_id in self.state_memory:
            del self.state_memory[episode_id]
        else:
            logger.warning(f"Episode_id {episode_id} not found in state_memory")

    @abstractmethod
    def initialize(self):
        pass
