import torch
import torch.nn as nn
from torch.utils.data import Dataset


class PreDataset(Dataset):
    def __init__(self, grouped_data):
        self.grouped_data = grouped_data

    def __len__(self):
        return len(self.grouped_data)

    def __getitem__(self, idx):
        group = self.grouped_data[idx]
        features = torch.tensor(group.iloc[:, 1:].values, dtype=torch.float)
        seq_length = len(features)
        return features, seq_length


def CollateforPretraining(batch):
    features = [item[0] for item in batch]
    seq_lengths = torch.tensor([item[1] for item in batch])
    padded_features = nn.utils.rnn.pad_sequence(features, batch_first=True, padding_value=0)
    max_length = padded_features.size(1)
    range_tensor = torch.arange(max_length).unsqueeze(0).expand(padded_features.size(0), -1)
    mask = range_tensor < seq_lengths.unsqueeze(1)
    return padded_features, mask


def min_max_scale(df, columns, col_min=0, col_max=1):
    for col in columns:
        original_min = df[col].min()
        original_max = df[col].max()

        df[col] = (df[col] - original_min) / (original_max - original_min) * (col_max - col_min) + col_min

        print(f"{col} - Min: {original_min}, Max: {original_max}")

    return df


def add_noise(data, noise_std):
    noisy_data = []
    for features, labels, seq_lengths in data:
        noise = torch.normal(mean=0, std=noise_std, size=features.size())
        noisy_features = features + noise
        noisy_data.append((noisy_features, labels, seq_lengths))
    return noisy_data
