import os
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset
from sklearn.preprocessing import MinMaxScaler
import pickle


class TabularDataset(Dataset):
    def __init__(self, path, input_dim, attrib_num, label_col=-1):
        df = pd.read_csv(path)
        self.y = df.iloc[:, label_col].values
        df = df.drop(df.columns[label_col], axis=1)
        scaler = MinMaxScaler(feature_range=(-1, 1))
        self.X = scaler.fit_transform(df)
        self.X = self.pad(self.X, input_dim)

    def pad(self, data, dim):
        padded = torch.zeros((data.shape[0], dim))
        padded[:, :data.shape[1]] = torch.tensor(data, dtype=torch.float32)
        return padded

    def __len__(self): return len(self.X)
    def __getitem__(self, idx): return self.X[idx], self.y[idx]


def show_all_parameters(model, name = "Model"):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"{name} | Total params: {total:,} | Trainable: {trainable:,}")


def load_transformer(path):
    with open(path, "rb") as f:
        transformer = pickle.load(f)
    return transformer


def load_processed_data(path='./preprocess/processed.csv'):
    print(f"Loading processed data from {path}...")
    return pd.read_csv(path).astype(np.float32).values


def extract_continuous_features(full_data, transformer_path='./preprocess/transformer/transformer.pkl'):
    transformer = load_transformer(transformer_path)
    output_info = transformer.output_info

    indices = []
    start = 0
    for item in output_info:
        dim, act = item
        if act == 'tanh':  # Continuous feature
            indices.extend(range(start, start + dim))
        start += dim

    cont_data = full_data[:, indices]
    return cont_data

def get_continuous_indices(output_info):
    """
    Return indices corresponding to continuous tanh blocks
    in the transformed tabular representation.
    """

    continuous_indices = []
    st = 0

    for dim, activation in output_info:
        dim = int(dim)
        ed = st + dim

        if activation == "tanh":
            continuous_indices.extend(
                range(st, ed)
            )

        st = ed

    return continuous_indices
