import numpy as np 
import zarr 

class DataLoader:
    """
    DataLoader is a class for loading data from a Zarr file.
    Attributes:
        description (str): Description of the module.
        path (str): Path to the Zarr file.
        batch_size (int): Size of the data batches to be loaded.
        zarr (zarr.hierarchy.Group): Zarr file object.
    Methods:
        __init__(path, batch_size=32):
            Initializes the DataLoader with the given path and batch size.
        get_joints_and_actions():
            Retrieves joints and actions data from the Zarr file.
            Returns:
                tuple: A tuple containing joints and actions arrays.
        get_episode_end_idx():
            Retrieves episode end indices from the Zarr file.
            Returns:
                numpy.ndarray: An array containing episode end indices.
    """
    
    def __init__(self, path, batch_size=32):
        self.description = "Module for Data Loading from Zarr"
        self.path = path
        self.batch_size = batch_size
        self.zarr = zarr.open(self.path, 'r')

    def get_joints_and_actions(self):
        joints = self.zarr['data/joints'][:]
        actions = self.zarr['data/action'][:]
        return joints, actions
    
    def get_episode_end_idx(self):
        episode_end_idx = self.zarr['meta/episode_end_idx'][:]
        return episode_end_idx