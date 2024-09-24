import numpy as np
import torch 


def adjacency_matrix_torch(num_joints, gripper_joint= True, gripper_link_all_joints = True):
    adjacency_matrix = torch.zeros(num_joints, num_joints)

    if gripper_joint:
        for i in range(num_joints-2): 
            adjacency_matrix[i, i+1] = 1
            adjacency_matrix[i+1, i] = 1

    else:
        for i in range(num_joints-1): 
            adjacency_matrix[i, i+1] = 1
            adjacency_matrix[i+1, i] = 1

    if gripper_joint:
        if gripper_link_all_joints:
            # Connect gripper to all other joints
            adjacency_matrix[-1, :-1] = 1
            adjacency_matrix[:-1, -1] = 1

        else:
            # Connect gripper to last joint
            adjacency_matrix[-1, -2] = 1
            adjacency_matrix[-2, -1] = 1
    
    return adjacency_matrix
