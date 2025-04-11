import torch

from preprocessing.type_utils import TrainingSample


class BaseDataset(torch.utils.data.Dataset):
    def __getitem__(self, idx: int) -> TrainingSample:
        """
        A dataset where each item is a dictionary with the following keys:
         - question_id : A unique question identifier
         - goal : An optional string with the question
         - action : The target action that should be performed given the observation
         - Observations : The observation provided to the agent
        """
        return TrainingSample()

    def get_scenes(self):
        """
        Return a list of all scenes in the dataset.
        Number of scenes may differ from length of the dataset.
        """
        raise NotImplementedError

    def subsample_scenes(self, scenes):
        """
        Keep only data from the specified scenes
        """
        raise NotImplementedError
