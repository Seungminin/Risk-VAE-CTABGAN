import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class Condvec:
    def __init__(self, data, output_info, device=None):
        self.device = device if device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = []
        self.interval = []
        self.n_col = 0
        self.n_opt = 0
        self.p_log_sampling = []
        self.p_sampling = []

        st = 0
        for item in output_info:
            if item[1] == 'tanh':
                st += item[0]
                continue
            elif item[1] == 'softmax':
                ed = st + item[0]
                self.model.append(np.argmax(data[:, st:ed], axis=-1))
                self.interval.append((self.n_opt, item[0]))
                self.n_col += 1
                self.n_opt += item[0]
                freq = np.sum(data[:, st:ed], axis=0)
                log_freq = np.log(freq + 1)
                self.p_log_sampling.append(torch.tensor(log_freq / np.sum(log_freq), dtype=torch.float32, device=self.device))
                self.p_sampling.append(torch.tensor(freq / np.sum(freq), dtype=torch.float32, device=self.device))
                st = ed

        self.interval = torch.tensor(self.interval, device=self.device)

    def sample_train(self, batch):
        if self.n_col == 0:
            return None

        vec = torch.zeros((batch, self.n_opt), dtype=torch.float32, device=self.device)
        mask = torch.zeros((batch, self.n_col), dtype=torch.float32, device=self.device)
        idx = torch.randint(0, self.n_col, (batch,), device=self.device)
        mask[torch.arange(batch, device=self.device), idx] = 1

        opt1prime = torch.empty(batch, dtype=torch.long, device=self.device)
        for i in range(batch):
            p = self.p_log_sampling[idx[i]] + 1e-6
            p = p / torch.sum(p)
            opt1prime[i] = torch.multinomial(p, 1).item()

        for i in range(batch):
            start = self.interval[idx[i], 0]
            vec[i, start + opt1prime[i]] = 1

        return vec, mask, idx, opt1prime

    def sample(self, batch):
        if self.n_col == 0:
            return None

        vec = torch.zeros((batch, self.n_opt), dtype=torch.float32, device=self.device)

        idx = torch.randint(0, self.n_col, (batch,), device=self.device)

        for i in range(batch):
            p = self.p_sampling[idx[i]] + 1e-6
            p = p / torch.sum(p)
            sampled = torch.multinomial(p, 1).item()
            start = self.interval[idx[i], 0]
            vec[i, start + sampled] = 1

        return vec.cpu().numpy()  # still returning numpy for now