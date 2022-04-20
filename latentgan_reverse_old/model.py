import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import time as t
import os
from latentgan_reverse.config import *


SAVE_PER_EPOCHS = save_per_epochs


class GeneratorEncoder(torch.nn.Module):
    def __init__(self, in_dim, hidden_dims, z_dim):
        super().__init__()
        dims = [in_dim] + hidden_dims
        modules = []
        for i in range(len(dims) - 1):
            modules.append(nn.Linear(dims[i], dims[i + 1]))
            modules.append(nn.ReLU(True))
        modules.append(nn.Linear(dims[-1], z_dim))
        self.main_module = nn.Sequential(*modules)

    def forward(self, x):
        return self.main_module(x)


class GeneratorDecoder(torch.nn.Module):
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


class GeneratorAE(object):
    def __init__(self):
        self.save_dir = os.path.join("experiments", model_name, model_params_subdir)
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)
        self.encoder = GeneratorEncoder(latent_size, hidden_dims_e, z_dim).cuda()
        self.decoder = GeneratorDecoder(z_dim, hidden_dims_g, latent_size).cuda() # pretrained generator
        for param in self.decoder.parameters():
            param.requires_grad = False

        self.learning_rate = learning_rate
        if batch_size == None:
            self.batch_size = 64
        else:
            self.batch_size = batch_size

        self.optimizer = optim.Adam(self.encoder.parameters(), lr=self.learning_rate)


    def forward(self, codes):
        return self.decoder(self.encoder(codes))


    def train(self, epochs, train_loader):
        t_begin = t.time()

        losses = []
        for epoch in range(1, epochs + 1):
            print("Epoch", epoch)
            for codes in train_loader:
                codes = codes.cuda()
                self.encoder.zero_grad()
                reconstructed_code = self.forward(codes)
                loss = F.mse_loss(reconstructed_code, codes)
                loss.backward()
                self.optimizer.step()
                losses.append(loss.item())

            if epoch % SAVE_PER_EPOCHS == 0:
                self.save_model(epoch)
                torch.save(
                    {
                        "loss": losses,
                    },
                    os.path.join("experiments", model_name, "Logs.pth")
                )

        t_end = t.time()
        print('Time of training: {}'.format((t_end - t_begin)))

        # Save the trained parameters
        self.save_model("latest")

        torch.save(
            {
                "loss": losses,
            },
            os.path.join("experiments", model_name, "Logs.pth")
        )


    def generate(self, num_codes=1, z=None):
        if z == None:
            z = torch.randn(num_codes, z_dim)
        samples = self.decoder(z.cuda()).detach()
        return samples


    def save_model(self, epoch):
        torch.save(self.encoder.state_dict(), os.path.join(self.save_dir, str(epoch) + "_e.pth"))
        torch.save(self.decoder.state_dict(), os.path.join(self.save_dir, str(epoch) + "_d.pth"))


    def load_decoder_from_generator(self, load_path):
        filepath_d = os.path.join(load_path)
        self.decoder.load_state_dict(torch.load(filepath_d))


    def load_model(self, load_dir, epoch):
        filepath_e = os.path.join(load_dir, str(epoch) + "_e.pth")
        filepath_d = os.path.join(load_dir, str(epoch) + "_d.pth")
        self.encoder.load_state_dict(torch.load(filepath_e))
        self.decoder.load_state_dict(torch.load(filepath_d))
        print('Encoder model loaded from {}.'.format(filepath_e))
        print('Decoder model loaded from {}-'.format(filepath_d))


    def get_infinite_batches(self, data_loader):
        while True:
            for real_latent_codes in data_loader:
                yield real_latent_codes


    def eval(self):
        self.encoder.eval()
        self.decoder.eval()

    # def generate_latent_walk(self, number):
    #     if not os.path.exists('interpolated_images/'):
    #         os.makedirs('interpolated_images/')

    #     number_int = 10
    #     # interpolate between twe noise(z1, z2).
    #     z_intp = torch.FloatTensor(1, 100, 1, 1)
    #     z1 = torch.randn(1, 100, 1, 1)
    #     z2 = torch.randn(1, 100, 1, 1)
    #     if self.cuda:
    #         z_intp = z_intp.cuda()
    #         z1 = z1.cuda()
    #         z2 = z2.cuda()

    #     z_intp = Variable(z_intp)
    #     real_latent_codes = []
    #     alpha = 1.0 / float(number_int + 1)
    #     print(alpha)
    #     for i in range(1, number_int + 1):
    #         z_intp.data = z1*alpha + z2*(1.0 - alpha)
    #         alpha += alpha
    #         fake_im = self.G(z_intp)
    #         fake_im = fake_im.mul(0.5).add(0.5) #denormalize
    #         real_latent_codes.append(fake_im.view(self.C,32,32).data.cpu())

    #     grid = utils.make_grid(real_latent_codes, nrow=number_int )
    #     utils.save_image(grid, 'interpolated_images/interpolated_{}.png'.format(str(number).zfill(3)))
    #     print("Saved interpolated real_latent_codes.")

