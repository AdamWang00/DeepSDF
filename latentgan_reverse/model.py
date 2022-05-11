import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import time as t
import os
from latentgan_reverse.config import *


class Generator(torch.nn.Module):
    def __init__(self, z_dim, hidden_dims, out_dim):
        super().__init__()
        dims = [z_dim] + hidden_dims
        modules = []
        for i in range(len(dims) - 1):
            modules.append(nn.Linear(dims[i], dims[i + 1]))
            modules.append(nn.ReLU(True))
        modules.append(nn.Linear(dims[-1], out_dim))
        self.main_module = nn.Sequential(*modules)

    def forward(self, x):
        return self.main_module(x)

    def generate(self, num_codes=1, z=None):
        if z == None:
            z = torch.randn(num_codes, z_dim)
        samples = self.forward(z.cuda()).detach()
        return samples



class GeneratorReverse(torch.nn.Module):
    def __init__(self, in_dim, hidden_dims, z_dim):
        super().__init__()
        dims = [in_dim] + hidden_dims
        modules = []
        for i in range(len(dims) - 1):
            modules.append(nn.Linear(dims[i], dims[i + 1]))
            modules.append(nn.ReLU(True))
        modules.append(nn.Linear(dims[-1], z_dim))
        self.main_module = nn.Sequential(*modules)

        self.save_dir = os.path.join(experiments_dir, model_name, model_params_subdir)
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)

        self.learning_rate = learning_rate
        if batch_size == None:
            self.batch_size = 64
        else:
            self.batch_size = batch_size

        self.optimizer = optim.Adam(self.parameters(), lr=self.learning_rate)


    def forward(self, x):
        return self.main_module(x)


    def train_(self, epochs, train_loader):
        t_begin = t.time()

        losses = []
        for epoch in range(1, epochs + 1):
            print("Epoch", epoch)
            for z, codes in train_loader:
                z = z.cuda()
                codes = codes.cuda()
                self.zero_grad()
                pred_z = self.forward(codes)
                loss = F.mse_loss(pred_z, z)
                loss.backward()
                self.optimizer.step()
                losses.append(loss.item())

            if epoch % save_per_epochs == 0:
                self.save_model(epoch)
                torch.save(
                    {
                        "loss": losses,
                    },
                    os.path.join(experiments_dir, model_name, "Logs.pth")
                )

        t_end = t.time()
        print('Time of training: {}'.format((t_end - t_begin)))

        # Save the trained parameters
        self.save_model("latest")

        torch.save(
            {
                "loss": losses,
            },
            os.path.join(experiments_dir, model_name, "Logs.pth")
        )

    
    def train_yield(self, epochs, train_loader):
        t_begin = t.time()

        losses = []
        for epoch in range(1, epochs + 1):
            print("Epoch", epoch)
            for z, codes in train_loader:
                z = z.cuda()
                codes = codes.cuda()
                self.zero_grad()
                pred_z = self.forward(codes)
                loss = F.mse_loss(pred_z, z)
                loss.backward()
                self.optimizer.step()
                losses.append(loss.item())

            if epoch % save_per_epochs == 0:
                self.save_model(epoch)
                torch.save(
                    {
                        "loss": losses,
                    },
                    os.path.join(experiments_dir, model_name, "Logs.pth")
                )
            
            yield

        t_end = t.time()
        print('Time of training: {}'.format((t_end - t_begin)))

        # Save the trained parameters
        self.save_model("latest")

        torch.save(
            {
                "loss": losses,
            },
            os.path.join(experiments_dir, model_name, "Logs.pth")
        )


    def generate(self, num_codes=1, z=None):
        if z == None:
            z = torch.randn(num_codes, z_dim)
        samples = self.decoder(z.cuda()).detach()
        return samples


    def save_model(self, epoch):
        torch.save(self.state_dict(), os.path.join(self.save_dir, str(epoch) + ".pth"))


    def load_model(self, load_dir, epoch):
        filepath = os.path.join(load_dir, str(epoch) + ".pth")
        self.load_state_dict(torch.load(filepath))
        print('GeneratorReverse model loaded from {}.'.format(filepath))