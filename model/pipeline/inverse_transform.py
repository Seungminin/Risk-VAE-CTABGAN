
import pickle
import pandas as pd
import numpy as np
import torch
import torch.nn.functional as F

class DummyTransformer:
    def __init__(self, transformer_path='./transformer.pkl'):
        with open(transformer_path, 'rb') as f:
            self.transformer = pickle.load(f)

    def inverse(self, data):
        return self.transformer.inverse_transform(data)

def inverse_transform(fake_data_np, transformer_path='./transformer.pkl', save_path='./preprocess/generated_restored.csv'):
    transformer = DummyTransformer(transformer_path)
    restored = transformer.inverse(fake_data_np)
    df = pd.DataFrame(restored)
    df.to_csv(save_path, index=False)
    print(f' Restored generated samples saved to {save_path}')

def apply_activate(data, output_info):
    data_t = []
    st = 0
    for item in output_info:
        if item[1] == 'tanh':
            ed = st + item[0]
            data_t.append(torch.tanh(data[:, st:ed]))
            st = ed
        elif item[1] == 'softmax':
            ed = st + item[0]
            data_t.append(F.gumbel_softmax(data[:, st:ed], tau=0.2))
            st = ed
    return torch.cat(data_t, dim=1)

