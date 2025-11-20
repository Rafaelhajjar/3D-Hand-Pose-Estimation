import matplotlib.pyplot as plt
import numpy as np
import torch

def vis_data_3d(kpts_3d, title=None):
    plt.figure(figsize=(4,4))
    ax = plt.axes(projection='3d')
    single_hand_index = np.array([[0,1,2,3,4],
                                [0,5,6,7,8],
                                [0,9,10,11,12],
                                [0,13,14,15,16],
                                [0,17,18,19,20]])
    color_dict = {0:'tab:blue', 1:'tab:orange', 2:'tab:green', 3:'tab:red', 4:'tab:purple'}
    # Plot each fingers with same color
    for i, finger_index in enumerate(single_hand_index):
        curr_finger_kpts = kpts_3d[finger_index]
        ax.scatter(curr_finger_kpts[:,0], curr_finger_kpts[:,1], curr_finger_kpts[:,2], color=color_dict[i])
        ax.plot3D(curr_finger_kpts[:,0], curr_finger_kpts[:,1], curr_finger_kpts[:,2], color=color_dict[i])
    # Adjust 3D viewing angle as needed
    ax.view_init(elev=-90, azim=180, roll=0)
    if title:
        ax.set_title(title)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
