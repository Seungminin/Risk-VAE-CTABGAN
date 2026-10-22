# model/sampler.py
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

class Sampler:
    def __init__(self, data, output_info, device=None):
        self.device = device if device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.data = torch.tensor(data, dtype=torch.float32, device=self.device)
        self.n = len(data)
        self.model = []

        st = 0
        for item in output_info:
            if item[1] == 'tanh':
                st += item[0]
                continue
            elif item[1] == 'softmax':
                ed = st + item[0]
                tmp = []
                for j in range(item[0]):
                    tmp.append(torch.nonzero(self.data[:, st + j]).squeeze(1))
                self.model.append(tmp)
                st = ed

    def sample(self, n, col, opt):
        if col is None:
            idx = torch.randint(0, self.n, (n,), device=self.device)
            return self.data[idx].cpu().numpy()

        sample_indices = []
        for c, o in zip(col, opt):
            try:
                candidates = self.model[c][o]
                if len(candidates) == 0:
                    raise IndexError("No candidates found.")
                idx = torch.randint(0, len(candidates), (1,), device=self.device).item()
                sample_indices.append(candidates[idx].item())
            except (IndexError, RuntimeError):
                fallback = torch.randint(0, self.n, (1,), device=self.device).item()
                sample_indices.append(fallback)

        return self.data[sample_indices].cpu().numpy()