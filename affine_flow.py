import torch
import numpy as np
from torch.distributions import MultivariateNormal, Uniform
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from time import sleep
import pandas as pd
from scipy import stats

class Net(nn.Module):
    def __init__(self, n_params, num_samples):
        super().__init__()
        self.N = n_params

        self.a = nn.Linear(1, self.N, bias=False)
        torch.exp_(self.a.weight.data)
        self.b = nn.Linear(1, self.N, bias=False)

        self.pz = MultivariateNormal(torch.zeros(self.N), torch.eye(self.N)) # latent distribution
        # self.px = Uniform(-1*torch.ones(self.N) + 3, 3*torch.ones(self.N)) # target distribution
        self.px = MultivariateNormal(torch.zeros(self.N) + 4, 3*torch.eye(self.N)) # target distribution

        self.num_samples = num_samples

    def forward(self, num_epochs=3):
        optimizer = optim.Adam(self.parameters(), lr=0.01)
        losses = []

        self.data = self.px.sample([self.num_samples]).reshape(self.N, -1) # sample from target distribution
        for epoch in tqdm(range(num_epochs)):

            log_jacob = -torch.log(self.a.weight).sum() # log of the determinant
            inverse = ((self.data - self.b.weight)/self.a.weight).mean(1) # the inverse of the T transformation

            # computes loss
            loss = 0
            for i in range(self.N):
                log_pz = self.pz.log_prob(inverse[i]) # log of the probability
                loss += -(log_pz + log_jacob)
            loss = (1/self.N) * loss
            losses.append(float(loss))

            if loss < 1e-4:
                print(f'ended at epoch: {epoch}')
                break

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            if (epoch + 1) % 25 == 0:
                print('loss:', round(float(loss), 3))

        return losses

    def sample(self, num_samples=100):
        t = self.px.sample([num_samples])
        og = self.pz.sample([num_samples])
        z = self.a.weight.view(-1) * og + self.b.weight.view(-1)
        return z.detach().numpy(), t.detach().numpy(), og.detach().numpy()

# %%
epochs = 1000
# %% 2 dimensional case
net2d = Net(n_params=2, num_samples=100_000)
losses2d = net2d(epochs)
data2d = net2d.sample(200)

modified = {
        ('latent', 'x'): data2d[0][:, 0], ('latent', 'y'): data2d[0][:, 1],
        ('target', 'x'): data2d[1][:, 0], ('target', 'y'): data2d[1][:, 1],
        ('original', 'x'): data2d[2][:, 0], ('original', 'y'): data2d[2][:, 1]
}

df2d = pd.DataFrame(modified)

sns.scatterplot(data=df2d['latent'], x='x', y='y', label='latent')
sns.scatterplot(data=df2d['target'], x='x', y='y', label='target')
sns.scatterplot(data=df2d['original'], x='x', y='y', label='original')

# plt.savefig(fname='2d-gaussian-approx')
# %% 1 dimensional case
net1d = Net(n_params=1, num_samples=10_00)
losses1d = net1d(epochs)
data1d = net1d.sample(400)

df1d = pd.DataFrame({'latent': data1d[0].reshape(-1), 'target': data1d[1].reshape(-1), 'original': data1d[2].reshape(-1)})

sns.histplot(data=df1d, bins='auto')
plt.xlim([-10, 10])

# plt.savefig(fname='1d-gaussian-approx')
# %%

plt.plot([i for i in range(len(losses1d))], losses1d, label='losses')
plt.plot([i for i in range(len(losses2d))], losses2d, label='losses2d')
