'''
Author: Uday Girish Maradana
Affiliation: WPI - Robotics Engineering
Description:
This script is used to infer the BoT-Mix Transformer Model from the Body Transformers paper
Current Stage:
Basic Implementation - No Support for Parallelism or Training
'''


import os 
import sys
import argparse
import torch    
from lib.data_loader import DataLoader
from lib.model import BoTMix
from lib.model_at_kvcache import BoTMixWithKVCache
from lib.helpers import adjacency_matrix_torch
import yaml 
import torch.nn as nn
from torchinfo import summary

# Import the Utils from abs path
sys.path.append("./")
from utils.logger import logger


# Prevent Python from genrating a .pyc file
sys.dont_write_bytecode = True


def inference_loop(model, joints, actions, episode_end_idx, device):
    """
    Perform inference on a model over multiple episodes and calculate the average mean squared error (MSE).

    Args:
        model (torch.nn.Module): The neural network model to evaluate.
        joints (list or np.ndarray): A list or array of joint states for each time step.
        actions (list or np.ndarray): A list or array of actions corresponding to each joint state.
        episode_end_idx (list of int): Indices indicating the end of each episode within the joints and actions lists.
        device (torch.device): The device (CPU or CUDA:x)to perform the computations on.

    Returns:
        float: The average mean squared error (MSE) over all episodes.
    """
    model.eval()
    total_mse = 0
    num_samples = 0
    start_idx = 0
    for end_idx in episode_end_idx:
        episode_joints = torch.tensor(joints[start_idx:end_idx]).float()
        episode_actions = torch.tensor(actions[start_idx:end_idx]).float()
        # Loop through the episode
        # Shift the joints and actions to device
        episode_joints = episode_joints.to(device)
        episode_actions = episode_actions.to(device)
        for i in range(len(episode_joints) - 1):
            input_state = episode_joints[i].unsqueeze(0)  # (1, 7)
            target_action = episode_actions[i + 1].unsqueeze(0)  # (1, 7)
            with torch.no_grad():
                predicted_action = model(input_state)  # (1, 7)
            mse = nn.functional.mse_loss(predicted_action, target_action)
            total_mse += mse.item()
            num_samples += 1
        start_idx = end_idx
    avg_mse = total_mse / num_samples
    return avg_mse

def train_loop(model, joints, actions, episode_end_idx, optimizer, device):
    """
    Trains the given model using the provided joint states and actions.

    Args:
        model (torch.nn.Module): The neural network model to be trained.
        joints (list or np.ndarray): A list or array of joint states.
        actions (list or np.ndarray): A list or array of actions corresponding to the joint states.
        episode_end_idx (list of int): Indices indicating the end of each episode.
        optimizer (torch.optim.Optimizer): The optimizer used for updating the model parameters.
        device (torch.device): The device (CPU or CUDA:x) on which the computations will be performed.

    Returns:
        float: The average loss over all training samples.
    """
    # Train function for Future Implementation
    model.train()
    total_loss = 0
    num_samples = 0
    start_idx = 0
    for end_idx in episode_end_idx:
        episode_joints = torch.tensor(joints[start_idx:end_idx]).float()
        episode_actions = torch.tensor(actions[start_idx:end_idx]).float()
        for i in range(len(episode_joints) - 1):
            input_state = episode_joints[i].unsqueeze(0)  # Shape - (1, 7)
            target_action = episode_actions[i + 1].unsqueeze(0)  # Shape - (1, 7)
            # Shift the joints and actions to device
            input_state = input_state.to(device)
            target_action = target_action.to(device)
            predicted_action = model(input_state)  # Shape - (1, 7)
            loss = nn.functional.mse_loss(predicted_action, target_action)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            num_samples += 1
        start_idx = end_idx
    avg_loss = total_loss / num_samples
    return avg_loss


def main(args):

    # Parse the Config File
    with open(args.config) as file:
        config = yaml.safe_load(file)

    # Load the Data
    data = DataLoader(config['data']['path'], config['data']['batch_size'])

    # Create the adjacency matrix
    adjacency_matrix = adjacency_matrix_torch(config['robot']['num_nodes'],
                                            config['robot']['gripper_joint'],
                                            config['robot']['gripper_link_to_all_joints'])
    # Create the Model based on the KV Cache Argument
    if args.kvcache:
        model = BoTMixWithKVCache(
            num_joints=config['robot']['num_nodes'], 
            d_model=config['bot_mix_transformer']['d_model'], 
            nhead=config['bot_mix_transformer']['num_heads'], 
            num_layers=config['bot_mix_transformer']['num_layers'], 
            feedforward_dim=config['bot_mix_transformer']['feedforward_dim'], 
            adjacency_matrix=adjacency_matrix,
            dropout=config['bot_mix_transformer']['dropout']
        )
    else:
        model = BoTMix(
            num_joints=config['robot']['num_nodes'], 
            d_model=config['bot_mix_transformer']['d_model'], 
            nhead=config['bot_mix_transformer']['num_heads'], 
            num_layers=config['bot_mix_transformer']['num_layers'], 
            feedforward_dim=config['bot_mix_transformer']['feedforward_dim'], 
            adjacency_matrix=adjacency_matrix,
            dropout=config['bot_mix_transformer']['dropout']
        )

    # Move the model to the device
    model = model.to(args.device)
    # Summary of the Model
    logger.info("=" * 20)
    logger.info("Model Summary: ")
    summary(model.to(args.device), (7,))
    logger.info("=" * 20)
    # Get joints, actions and episode end indices
    joints, actions = data.get_joints_and_actions()
    episode_end_idx = data.get_episode_end_idx()



    # Print the episode end indices
    # print("Joints Shape: ", joints.shape)
    # print("Actions Shape: ", actions.shape)
    # print("Episode End Indices: ", episode_end_idx)
    # Inference Loop 
    avg_mse = inference_loop(model, joints, actions, episode_end_idx, args.device)
    logger.info("TEST INFERENCE LOOP Average MSE: " + str(round(avg_mse, 6)))



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='config/pipeline_config.yaml')
    parser.add_argument('--device', type=str, default=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    parser.add_argument('--kvcache', type=bool, default=False)
    args = parser.parse_args()
    main(args)


