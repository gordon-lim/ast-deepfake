# -*- coding: utf-8 -*-
# @Time    : 6/19/21 12:23 AM
# @Author  : Yuan Gong
# @Affiliation  : Massachusetts Institute of Technology
# @Email   : yuangong@mit.edu
# @File    : dataloader.py

# modified from:
# Author: David Harwath
# with some functions borrowed from https://github.com/SeanNaren/deepspeech.pytorch

import csv
import json
import torchaudio
import numpy as np
import torch
import torch.nn.functional
from torch.utils.data import Dataset
import random
import pandas as pd
import librosa
import os
import matplotlib.pyplot as plt


def preemphasis(signal, coeff=0.97):
    """perform preemphasis on the input signal.

    :param signal: The signal to filter.
    :param coeff: The preemphasis coefficient. 0 is none, default 0.97.
    :returns: the filtered signal.
    """
    return np.append(signal[0], signal[1:]-coeff*signal[:-1])


def plot_fbank(fbank, name, index=0):
    plt.figure(figsize=(10, 4))
    plt.imshow(fbank.T, cmap='hot', interpolation='nearest')
    plt.colorbar()
    plt.title('fbank')
    plt.tight_layout()
    plt.savefig(f'{name}_{index}.png')
    plt.show()


class AudiosetDataset(Dataset):
    def __init__(self, dataset_csv_file, audio_conf, label_csv=None):
        """
        Dataset that manages audio recordings
        :param audio_conf: Dictionary containing the audio loading and preprocessing settings
        :param dataset_json_file
        """
        self.datapath = dataset_csv_file
        self.count = 0

        # Load the CSV file
        try:
            df = pd.read_csv(dataset_csv_file)

            # Create the desired dictionary structure
            data_dict = {
                "data": [{"path": row["File Path"], "label": row["Label"]} for _, row in df.iterrows()]
            }

        except FileNotFoundError:
            print(
                f"File not found: {dataset_csv_file}. Please check the file path and try again.")
        except Exception as e:
            print(f"An error occurred: {e}")

        self.name = dataset_csv_file.split('/')[-1].split('.')[0]
        self.saved_data = []
        self.data = data_dict['data']
        self.audio_conf = audio_conf
        print(
            '---------------the {:s} dataloader---------------'.format(self.audio_conf.get('mode')))
        self.melbins = self.audio_conf.get('num_mel_bins')
        self.freqm = self.audio_conf.get('freqm')
        self.timem = self.audio_conf.get('timem')
        print('now using following mask: {:d} freq, {:d} time'.format(
            self.audio_conf.get('freqm'), self.audio_conf.get('timem')))
        self.mixup = self.audio_conf.get('mixup')
        print('now using mix-up with rate {:f}'.format(self.mixup))
        self.dataset = self.audio_conf.get('dataset')
        print('now process ' + self.dataset)
        # dataset spectrogram mean and std, used to normalize the input
        self.norm_mean = self.audio_conf.get('mean')
        self.norm_std = self.audio_conf.get('std')
        # Delta and Delta-Delta normalization stats
        self.delta_mean = self.audio_conf.get('delta_mean', 0.0)
        self.delta_std = self.audio_conf.get('delta_std', 1.0)
        self.delta_delta_mean = self.audio_conf.get('delta_delta_mean', 0.0)
        self.delta_delta_std = self.audio_conf.get('delta_delta_std', 1.0)
        # skip_norm is a flag that if you want to skip normalization to compute the normalization stats using src/get_norm_stats.py, if Ture, input normalization will be skipped for correctly calculating the stats.
        # set it as True ONLY when you are getting the normalization stats.
        self.skip_norm = self.audio_conf.get(
            'skip_norm') if self.audio_conf.get('skip_norm') else False
        if self.skip_norm:
            print(
                'now skip normalization (use it ONLY when you are computing the normalization stats).')
        else:
            print('use dataset mean {:.3f} and std {:.3f} to normalize the input.'.format(
                self.norm_mean, self.norm_std))
        # if add noise for data augmentation
        self.noise = self.audio_conf.get('noise')
        if self.noise == True:
            print('now use noise augmentation')

        self.label_num = 2
        print('number of classes is {:d}'.format(self.label_num))

    def _wav2fbank(self, filename, filename2=None):
        if filename2 == None:
            waveform, sr = librosa.load(filename, sr=None)
            waveform = waveform - waveform.mean()
        # mixup
        else:
            waveform1, sr = torchaudio.load(filename)
            waveform2, _ = torchaudio.load(filename2)

            waveform1 = waveform1 - waveform1.mean()
            waveform2 = waveform2 - waveform2.mean()

            if waveform1.shape[1] != waveform2.shape[1]:
                if waveform1.shape[1] > waveform2.shape[1]:
                    # padding
                    temp_wav = torch.zeros(1, waveform1.shape[1])
                    temp_wav[0, 0:waveform2.shape[1]] = waveform2
                    waveform2 = temp_wav
                else:
                    # cutting
                    waveform2 = waveform2[0, 0:waveform1.shape[1]]

            # sample lambda from uniform distribution
            # mix_lambda = random.random()
            # sample lambda from beta distribtion
            mix_lambda = np.random.beta(10, 10)

            mix_waveform = mix_lambda * waveform1 + \
                (1 - mix_lambda) * waveform2
            waveform = mix_waveform - mix_waveform.mean()

        # Convert waveform from NumPy array to PyTorch tensor
        waveform = torch.tensor(waveform, dtype=torch.float32)

        # Add a batch dimension if required (for single-channel audio)
        if len(waveform.shape) == 1:
            waveform = waveform.unsqueeze(0)  # Shape: [1, num_samples]

        fbank = torchaudio.compliance.kaldi.fbank(waveform, htk_compat=True, sample_frequency=sr, use_energy=False,
                                                  window_type='hanning', num_mel_bins=self.melbins, dither=0.0, frame_shift=10)

        target_length = self.audio_conf.get('target_length')
        n_frames = fbank.shape[0]

        p = target_length - n_frames

        # cut and pad
        if p > 0:
            m = torch.nn.ZeroPad2d((0, 0, 0, p))
            fbank = m(fbank)
        elif p < 0:
            fbank = fbank[0:target_length, :]

        if filename2 == None:
            return fbank, 0
        else:
            return fbank, mix_lambda

    def __getitem__(self, index):
        """
        returns: image, audio, nframes
        where image is a FloatTensor of size (3, H, W)
        audio is a FloatTensor of size (N_freq, N_frames) for spectrogram, or (N_frames) for waveform
        nframes is an integer
        """
        # do mix-up for this sample (controlled by the given mixup rate)
        if random.random() < self.mixup:
            pass
        # if not do mixup
        else:
            datum = self.data[index]
            label_indices = np.zeros(2)
            fbank, mix_lambda = self._wav2fbank(datum['path'])
            for label_str in datum['label'].split(','):
                if label_str == "spoof":
                    label_indices[0] = 1
                elif label_str == "bonafide":
                    label_indices[1] = 1
                else:
                    print(f"error: unknown label {label_str}")

            label_indices = torch.FloatTensor(label_indices)
        # plot_fbank(fbank, "before", index)

        # SpecAug, not do for eval set
        freqm = torchaudio.transforms.FrequencyMasking(self.freqm)
        timem = torchaudio.transforms.TimeMasking(self.timem)
        fbank = torch.transpose(fbank, 0, 1)
        # this is just to satisfy new torchaudio version, which only accept [1, freq, time]
        fbank = fbank.unsqueeze(0)
        delta_transform = torchaudio.transforms.ComputeDeltas(win_length=5)
        delta = delta_transform(fbank)
        delta_delta = delta_transform(delta)
        fbank = torch.cat([fbank, delta, delta_delta], dim=0)  # Shape: [3, time, freq]
        if self.freqm != 0:
            fbank = freqm(fbank)
        if self.timem != 0:
            fbank = timem(fbank)
        # squeeze it back, it is just a trick to satisfy new torchaudio version
        # fbank = fbank.squeeze(0)
        fbank = torch.transpose(fbank, 1, 2)
        # normalize the input for both training and test
        if not self.skip_norm:
            fbank[0] = (fbank[0] - self.norm_mean) / (self.norm_std * 2)
            fbank[1] = (fbank[1] - self.delta_mean) / (self.delta_std * 2)  # Delta
            fbank[2] = (fbank[2] - self.delta_delta_mean) / (self.delta_delta_std * 2)  # Delta-Delta
        # skip normalization the input if you are trying to get the normalization stats.
        else:
            pass

        if self.noise == True:
            fbank = fbank + \
                torch.rand(fbank.shape[0], fbank.shape[1]
                           ) * np.random.rand() / 10
            fbank = torch.roll(fbank, np.random.randint(-10, 10), 0)

        mix_ratio = min(mix_lambda, 1-mix_lambda) / \
            max(mix_lambda, 1-mix_lambda)

        spectro = {'fbank': fbank, 'label': label_indices}
        # plot_fbank(fbank, "after", index)
        if self.count == 100:
            print("its working")
        if (self.count % 10000 == 0):
            print(f"Processed {self.count} samples")
        self.saved_data.append(spectro)
        self.count += 1

        # At the end of the dataset iteration (after the DataLoader is done)
        if index == len(self.data) - 1:

            base = os.getcwd()
            output_dir = base + '/datafiles/' + f'{self.name}.pt'
            print(f"Saving all data to {output_dir}")
            torch.save(self.saved_data, output_dir)
            print(
                f"All data saved to {os.path.join(output_dir, 'all_data.pt')}")

        # the output fbank shape is [time_frame_num, frequency_bins], e.g., [1024, 128]
        # print(f"fbank shape: {fbank.shape}, label_indices: {label_indices}")
        return fbank, label_indices

    def __len__(self):
        return len(self.data)
