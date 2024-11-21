import os
import numpy as np
import torch
import new_dataloader


def main():
    base = os.getcwd()
    data_path = 'train_la_data.csv'

    # Set skip_norm as True only when computing normalization stats
    audio_conf = {
        'num_mel_bins': 128,
        'target_length': 512,
        'freqm': 24,
        'timem': 96,
        'mean': -3.8588,
        'std': 5.2843,
        'mixup': 0,
        'skip_norm': True,
        'mode': 'train',
        'dataset': 'asvspoof2021',
        'noise': False
    }

    # Use a smaller batch size if memory is limited
    train_loader = torch.utils.data.DataLoader(
        new_dataloader.AudiosetDataset(data_path, audio_conf=audio_conf),
        batch_size=24,  # Reduced batch size
        shuffle=False, num_workers=8, pin_memory=True
    )

    # Initialize variables for running mean and variance calculation
    total_sum = 0
    total_sum_sq = 0
    total_samples = 0

    for i, (audio_input, labels) in enumerate(train_loader):
        # print(f"Batch {i}: audio_input.shape = {audio_input.shape}")

        # Calculate batch statistics
        batch_sum = torch.sum(audio_input)
        batch_sum_sq = torch.sum(audio_input ** 2)
        batch_count = audio_input.numel()

        # Update running totals
        total_sum += batch_sum.item()
        total_sum_sq += batch_sum_sq.item()
        total_samples += batch_count

        # Print progress every 1000 batches
        if i % 1000 == 0:
            print(f"Processed {i} batches...")

    # Compute overall mean and std
    overall_mean = total_sum / total_samples
    overall_std = ((total_sum_sq / total_samples) - overall_mean ** 2) ** 0.5

    print(f"Overall Mean: {overall_mean:.4f}, Overall Std: {overall_std:.4f}")


if __name__ == '__main__':
    main()
