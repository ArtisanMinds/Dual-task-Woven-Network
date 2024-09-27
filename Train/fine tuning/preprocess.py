import torch
import torch.nn as nn
from torch.utils.data import Dataset


class Dataset(Dataset):
    def __init__(self, grouped_data):
        self.grouped_data = grouped_data

    def __len__(self):
        return len(self.grouped_data)

    def __getitem__(self, idx):
        group = self.grouped_data[idx]
        features = torch.tensor(group.iloc[:, 1:-1].values, dtype=torch.float)
        labels = torch.tensor(group.iloc[:, -1].values, dtype=torch.long)-1  # ignore 0
        seq_length = len(features)
        return features, labels, seq_length

class Dataset_test(Dataset):
    def __init__(self, grouped_data):
        self.grouped_data = grouped_data

    def __len__(self):
        return len(self.grouped_data)

    def __getitem__(self, idx):
        group = self.grouped_data[idx]
        features = torch.tensor(group.iloc[:, 1:-1].values, dtype=torch.float)
        labels = torch.tensor(group.iloc[:, -1].values, dtype=torch.long)  # for test
        seq_length = len(features)
        return features, labels, seq_length


def CollateforTraining(batch):
    features = [item[0] for item in batch]
    labels = [item[1] for item in batch]
    seq_lengths = torch.tensor([item[2] for item in batch])
    padded_features = nn.utils.rnn.pad_sequence(features, batch_first=True, padding_value=0)
    padded_labels = nn.utils.rnn.pad_sequence(labels, batch_first=True, padding_value=-1)
    max_length = padded_features.size(1)
    range_tensor = torch.arange(max_length).unsqueeze(0).expand(padded_features.size(0), -1)
    mask = range_tensor < seq_lengths.unsqueeze(1)
    return padded_features, padded_labels, mask

def min_max_scale(df, columns, global_mins, global_maxs, col_min=0, col_max=1):
    for col in columns:
        original_min = global_mins[col]
        original_max = global_maxs[col]

        df[col] = (df[col] - original_min) / (original_max - original_min) * (col_max - col_min) + col_min

    return df


def add_noise(data, noise_std):
    noisy_data = []
    for features, labels, seq_lengths in data:
        noise = torch.normal(mean=0, std=noise_std, size=features.size())
        noisy_features = features + noise
        noisy_data.append((noisy_features, labels, seq_lengths))
    return noisy_data
