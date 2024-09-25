import time
import torch
import torch.nn as nn
from lib.model import BoTMix
from lib.model_kvcache import BoTMixWithKVCache
from utils.logger import logger

def compare_models_performance(config, adjacency_matrix, joints,  actions, episode_end_idx, device):
    """
    Compare the performance of two models: one with KV Cache and one without.
    Args:
        config (dict): Configuration dictionary containing model and robot parameters.
        adjacency_matrix (torch.Tensor): Adjacency matrix representing the robot's joint connections.
        joints (list): List of joint states for each episode.
        actions (list): List of actions corresponding to the joint states.
        episode_end_idx (list): List of indices indicating the end of each episode.
        device (torch.device): Device to run the models on (e.g., 'cpu' or 'cuda').
    Returns:
        None: Logs the average MSE and inference time for both models, and the speed improvement.
    """
    
    model_no_kvcache = BoTMix(
        num_joints=config['robot']['num_nodes'],
        d_model=config['bot_mix_transformer']['d_model'],
        nhead=config['bot_mix_transformer']['num_heads'],
        num_layers=config['bot_mix_transformer']['num_layers'],
        dim_feedforward=config['bot_mix_transformer']['feedforward_dim'],
        adjacency_matrix=adjacency_matrix,
        dropout=config['bot_mix_transformer']['dropout'],
        device=device
    ).to(device)

    model_with_kvcache = BoTMixWithKVCache(
        num_joints=config['robot']['num_nodes'],
        d_model=config['bot_mix_transformer']['d_model'],
        nhead=config['bot_mix_transformer']['num_heads'],
        num_layers=config['bot_mix_transformer']['num_layers'],
        dim_feedforward=config['bot_mix_transformer']['feedforward_dim'],
        adjacency_matrix=adjacency_matrix,
        dropout=config['bot_mix_transformer']['dropout'],
        device=device
    ).to(device)

    # Function to run inference for a single model
    def run_inference(model, use_kvcache):
        model.eval()
        total_mse = 0
        num_samples = 0
        start_idx = 0
        kv_cache = None
        start_time = time.time()
        
        for end_idx in episode_end_idx:
            episode_joints = torch.tensor(joints[start_idx:end_idx]).float().to(device)
            episode_actions = torch.tensor(actions[start_idx:end_idx]).float().to(device)
            for i in range(len(episode_joints) - 1):
                input_state = episode_joints[i].unsqueeze(0).unsqueeze(-1)  # Shape: (1, 7, 1)
                target_action = episode_actions[i + 1].unsqueeze(0)  # Shape: (1, 7)
                with torch.no_grad():
                    if use_kvcache:
                        predicted_action, kv_cache = model(input_state, kv_cache)
                        predicted_action = predicted_action.squeeze(-1)
                    else:
                        predicted_action = model(input_state).squeeze(-1)
                mse = nn.functional.mse_loss(predicted_action, target_action)
                total_mse += mse.item()
                num_samples += 1
            start_idx = end_idx
        
        end_time = time.time()
        avg_mse = total_mse / num_samples
        total_time = end_time - start_time
        return avg_mse, total_time

    # Run inference for both models
    logger.info("Running inference without KV Cache...")
    avg_mse_no_kvcache, time_no_kvcache = run_inference(model_no_kvcache, False)
    
    logger.info("Running inference with KV Cache...")
    avg_mse_with_kvcache, time_with_kvcache = run_inference(model_with_kvcache, True)

    # Log results
    logger.info(f"Model without KV Cache - Avg MSE: {avg_mse_no_kvcache:.6f}, Time: {time_no_kvcache:.4f} seconds")
    logger.info(f"Model with KV Cache - Avg MSE: {avg_mse_with_kvcache:.6f}, Time: {time_with_kvcache:.4f} seconds")
    logger.info(f"Speed improvement: {(time_no_kvcache - time_with_kvcache) / time_no_kvcache * 100:.2f}%")