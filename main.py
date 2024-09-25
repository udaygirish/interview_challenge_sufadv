'''
Author: Uday Girish Maradana
Affiliation: WPI - Robotics Engineering
Description: This script is used to infer the BoT-Mix Transformer Model from the Body Transformers paper
Current Stage: Basic Implementation - No Support for Parallelism or Training
'''

import os
import sys
import argparse
import torch
import yaml
import torch.nn as nn
from torchinfo import summary

# Add the project root to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from lib.data_loader import DataLoader
from lib.model import BoTMix
from lib.model_kvcache import BoTMixWithKVCache
from lib.helpers import adjacency_matrix_torch
from utils.logger import logger

# Prevent Python from generating a .pyc file
sys.dont_write_bytecode = True

def inference_loop(model, joints, actions, episode_end_idx, device):
    """
    Perform inference on a model over multiple episodes and calculate the average mean squared error (MSE).
    """
    model.eval()
    total_mse = 0
    num_samples = 0
    start_idx = 0
    kv_cache = None
    for end_idx in episode_end_idx:
        episode_joints = torch.tensor(joints[start_idx:end_idx]).float().to(device)
        episode_actions = torch.tensor(actions[start_idx:end_idx]).float().to(device)
        for i in range(len(episode_joints) - 1):
            input_state = episode_joints[i].unsqueeze(0).unsqueeze(-1)  # Shape: (1, 7, 1)
            target_action = episode_actions[i + 1].unsqueeze(0)  # Shape: (1, 7)
            with torch.no_grad():
                if isinstance(model, BoTMixWithKVCache):
                    predicted_action, kv_cache = model(input_state, kv_cache)
                    predicted_action = predicted_action.squeeze(-1)
                else:
                    predicted_action = model(input_state).squeeze(-1)
            mse = nn.functional.mse_loss(predicted_action, target_action)
            total_mse += mse.item()
            num_samples += 1
        start_idx = end_idx
    avg_mse = total_mse / num_samples
    return avg_mse

def main(args):
    """
    Main function to execute the model training and inference pipeline.

    Args:
        args (Namespace): Command-line arguments containing configuration file path, device, and KV cache flag.

    Workflow:
        1. Parse the configuration file to load model and data parameters.
        2. Load the dataset using DataLoader.
        3. Create the adjacency matrix for the robot's joints.
        4. Initialize the model (BoTMix or BoTMixWithKVCache) based on the KV cache argument.
        5. Move the model to the specified device (CPU/GPU).
        6. Log the model summary.
        7. Retrieve joints, actions, and episode end indices from the dataset.
        8. Perform inference loop and log the average Mean Squared Error (MSE).

    Returns:
        None
    """
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
    model_class = BoTMixWithKVCache if args.kvcache else BoTMix
    model = model_class(
        num_joints=config['robot']['num_nodes'],
        d_model=config['bot_mix_transformer']['d_model'],
        nhead=config['bot_mix_transformer']['num_heads'],
        num_layers=config['bot_mix_transformer']['num_layers'],
        dim_feedforward=config['bot_mix_transformer']['feedforward_dim'],
        adjacency_matrix=adjacency_matrix,
        dropout=config['bot_mix_transformer']['dropout'],
        device=args.device
    )

    # Move the model to the device
    model = model.to(args.device)

    # Summary of the Model
    logger.info("=" * 20)
    logger.info("Model Summary: ")
    summary(model, input_size=(1, 7, 1))  # Assuming input shape is (batch_size, num_joints, 1)
    logger.info("=" * 20)

    # Get joints, actions and episode end indices
    joints, actions = data.get_joints_and_actions()
    episode_end_idx = data.get_episode_end_idx()

    # Inference Loop 
    avg_mse = inference_loop(model, joints, actions, episode_end_idx, args.device)
    logger.info(f"TEST INFERENCE LOOP Average MSE: {avg_mse:.6f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='config/pipeline_config.yaml')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--kvcache', action='store_true')
    args = parser.parse_args()
    main(args)