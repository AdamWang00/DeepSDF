import matplotlib.pyplot as plt
import numpy as np
import torch
import os
from latentvae.config import *

def running_mean(x, N):
    cumsum = np.cumsum(np.insert(x, 0, 0))
    return (cumsum[N:] - cumsum[:-N]) / float(N)

logs = torch.load(os.path.join("experiments", model_name, "Logs.pth"))

num_iters = len(logs["loss"])

smoothed_loss_41 = running_mean(logs["loss"], 41)

fig, ax = plt.subplots()

ax.plot(
    np.arange(num_iters),
    logs["loss"],
    "#82c6eb",
    logs["kld_loss"],
    "#6759ff",
    smoothed_loss_41,
    "#2a9edd",
)

ax.grid()
# plt.show()
plt.savefig(f"logs_{model_name}.png")