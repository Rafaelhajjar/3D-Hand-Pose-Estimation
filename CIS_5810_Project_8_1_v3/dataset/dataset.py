import copy
import json
import os

import cv2
import imageio
import numpy as np
import torch
from torch.utils.data import Dataset


class ego4dDataset(Dataset):
    """
    Implementation of Ego-Exo4D dataset for hand pose estimation. 
    Return cropped hand image, ground truth heatmap and valid joint flag as output.
    """
    def __init__(self, data_root, split, normalize=True, replace_nan=True):
        """
        Args:
            data_root: Directory of ground truth annotation file
            split: current split of the dataset
            normalize: whether apply normalization to 2D and 3D coordinates
            replace_nan: whether replace all NaN values to zero
        """
        self.split = split
        assert self.split in ["train", "val", "test"], f"Invalid split: {self.split}. Can only be [train, val, test]"
        gt_anno_path = os.path.join(data_root, f"ego_pose_gt_anno_{self.split}.json")
        self.db = self.load_all_data(gt_anno_path)
        self.apply_normalize = normalize
        self.replace_nan = replace_nan
        # Normalization parameter
        self.mean_2d = np.array([282.57942706, 337.41327088])
        self.std_2d = np.array([65.29158, 50.37797])
        self.mean_3d = np.array([-0.04980424, 0.00384788, 0.07262705])
        self.std_3d = np.array([0.04344871, 0.05391886, 0.05025425])


    def __getitem__(self, idx):
        """
        Return 2D gt kpts, offseted 3D gt kpts in camera coordinate system, weight (boolean flag), metadata.
        Apply normalization as needed.
        """
        curr_db = copy.deepcopy(self.db[idx])

        # 1. Load 2d and 3d joint coordinates
        joints_2d = curr_db["joints_2d"]
        joints_3d = curr_db["joints_3d"]

        # 2. Normalize joint coordinates; offset 3D gt hand joints
        joints_2d = self.normalize_2d(joints_2d) if self.apply_normalize else joints_2d
        joints_2d = torch.from_numpy(joints_2d).to(torch.float32)

        joints_3d -= joints_3d[0]
        joints_3d = self.normalize_3d(joints_3d) if self.apply_normalize else joints_3d
        joints_3d = torch.from_numpy(joints_3d).to(torch.float32)

        # 3. Replace NaN with zero
        if self.replace_nan:
            joints_2d = torch.nan_to_num(joints_2d)
            joints_3d = torch.nan_to_num(joints_3d)

        # 4. Generate valid joints flag
        weight = torch.from_numpy(curr_db["valid_flag"])

        # keep metadata info for debugging
        meta = curr_db["metadata"]

        return joints_2d, joints_3d, weight, meta


    def __len__(self):
        return len(self.db)


    def load_all_data(self, gt_anno_path):
        """
        Store each valid hand's annotation per frame separately as a dictionary 
        with following key:
            - joints_2d: (N,2)
            - joints_3d: (N,3)
            - valid_flag: (N,)
            - metadata
        """
        # Load ground truth annotation
        gt_anno = json.load(open(gt_anno_path))

        # Load gt annotation
        all_frame_anno = []
        for _, curr_take_anno in gt_anno.items():
            for _, curr_f_anno in curr_take_anno.items():
                for hand_order in ["right", "left"]:
                    single_hand_anno = {}
                    if len(curr_f_anno[f"{hand_order}_hand_2d"]) != 0:
                        single_hand_anno["joints_2d"] = np.array(
                            curr_f_anno[f"{hand_order}_hand_2d"]
                        )
                        single_hand_anno["joints_3d"] = np.array(
                            curr_f_anno[f"{hand_order}_hand_3d"]
                        )
                        single_hand_anno["valid_flag"] = np.array(
                            curr_f_anno[f"{hand_order}_hand_valid"]
                        )
                        single_hand_anno["metadata"] = curr_f_anno["metadata"]
                        all_frame_anno.append(single_hand_anno)
        return all_frame_anno


    def normalize_2d(self, kpts):
        return (kpts - self.mean_2d) / self.std_2d

    def inv_normalize_2d(self, kpts):
        return kpts * self.std_2d + self.mean_2d
    
    def normalize_3d(self, kpts):
        return (kpts - self.mean_3d) / self.std_3d

    def inv_normalize_3d(self, kpts):
        return kpts * self.std_3d + self.mean_3d
