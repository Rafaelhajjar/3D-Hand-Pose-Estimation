import logging
import time
from pathlib import Path
import numpy as np
import math
import torchvision
import cv2
import os
from tqdm import tqdm


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count if self.count != 0 else 0


def create_logger(log_output_dir, phase="train"):
    os.makedirs(log_output_dir, exist_ok=True)
    time_str = time.strftime("%Y-%m-%d-%H-%M")
    final_output_dir = Path(log_output_dir) / (time_str)
    # set up logger
    if not final_output_dir.exists():
        print("=> creating {}".format(final_output_dir))
        final_output_dir.mkdir()

    log_file = "{}_{}.log".format(time_str, phase)
    final_log_file = final_output_dir / log_file
    head = "%(asctime)-15s %(message)s"
    logging.basicConfig(filename=str(final_log_file), format=head)
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    console = logging.StreamHandler()
    logging.getLogger("").addHandler(console)

    return logger, str(final_output_dir), time_str


def normalize_stats(dataset):
    # Compute joints normalization parameter
    all_kpts_2d = []
    all_kpts_3d = []
    for kpts_2d, kpts_3d, _, _ in tqdm(dataset, total=len(dataset)):
        all_kpts_2d.append(kpts_2d.numpy())
        all_kpts_3d.append(kpts_3d.numpy())
    all_kpts_2d = np.concatenate(all_kpts_2d, axis=0)
    all_kpts_3d = np.concatenate(all_kpts_3d, axis=0)
    # Calculate mean and std
    mean_2d = np.nanmean(all_kpts_2d, axis=0)
    std_2d = np.nanstd(all_kpts_2d, axis=0)
    mean_3d = np.nanmean(all_kpts_3d, axis=0)
    std_3d = np.nanstd(all_kpts_3d, axis=0)
    print(f"2d: mean: {mean_2d}\t std: {std_2d}")
    print(f"3d: mean: {mean_3d}\t std: {std_3d}")