import numpy as np 
import zarr 

class DataLoader:
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