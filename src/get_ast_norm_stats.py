# -*- coding: utf-8 -*-
# @Time    : 8/4/21 4:30 PM
# @Author  : Yuan Gong
# @Affiliation  : Massachusetts Institute of Technology
# @Email   : yuangong@mit.edu
# @File    : get_norm_stats.py

# This is a sample code to get normalization stats for input spectrogram

import os
import numpy as np
import torch
import dataloader


def main():
    base = os.getcwd()
    data_path = base + '/combined_df_data.csv'

    # Set skip_norm as True only when computing normalization stats
    audio_conf = {
        'num_mel_bins': 128,
        'target_length': 512,
        'freqm': 0,  # 24,
        'timem': 0,  # 96,
        'mixup': 0,
        'skip_norm': True,
        'mode': 'train',
        'dataset': 'asvspoof2021',
        'noise': False
    }

    train_loader = torch.utils.data.DataLoader(
        dataloader.AudiosetDataset(data_path, audio_conf=audio_conf),
        batch_size=24, shuffle=False, num_workers=8, pin_memory=True
    )

    mean = []
    std = []

    for i, (audio_input, labels) in enumerate(train_loader):
        cur_mean = torch.mean(audio_input)
        cur_std = torch.std(audio_input)
        mean.append(cur_mean.item())  # Convert tensors to Python scalars
        std.append(cur_std.item())
        # print(f"Batch {i}: Mean = {cur_mean:.4f}, Std = {cur_std:.4f}")

    print(
        f"Overall Mean: {np.mean(mean):.4f}, Overall Std: {np.mean(std):.4f}")


if __name__ == '__main__':
    main()
