import matplotlib.pyplot as plt
import numpy as np
import torch
import os
from latentgan.config import *

def running_mean(x, N):
    cumsum = np.cumsum(np.insert(x, 0, 0))
    return (cumsum[N:] - cumsum[:-N]) / float(N)

logs = torch.load(os.path.join("experiments", model_name, "Logs.pth"))

num_iters = len(logs["loss_g"])

smoothed_loss_g_41 = running_mean(logs["loss_g"], 41)
smoothed_loss_d_41 = running_mean(logs["loss_d"], 41)

fig, ax = plt.subplots()

ax.plot(
    np.arange(num_iters),
    logs["loss_g"],
    "#82c6eb",
    smoothed_loss_g_41,
    "#2a9edd",
    np.arange(num_iters),
    logs["loss_d"],
    "#ebe652",
    smoothed_loss_d_41,
    "#aba612",
)

ax.grid()
# plt.show()
plt.savefig("logs.png")