import torch
import torch.nn as nn
import torch.optim as optim
import time as t
import os
from torch.autograd import Variable
from torch import autograd
from latentgan.config import *


SAVE_PER_ITERS = save_per_iters


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


class Discriminator(torch.nn.Module):
    def __init__(self, in_dim, hidden_dims, out_dim=1):
        super().__init__()
        dims = [in_dim] + hidden_dims
        modules = []
        for i in range(len(dims) - 1):
            modules.append(nn.Linear(dims[i], dims[i + 1]))
            modules.append(nn.ReLU(True))
        modules.append(nn.Linear(dims[-1], out_dim))
        self.main_module = nn.Sequential(*modules)

    def forward(self, x):
        return self.main_module(x)


class WGAN_GP(object):
    def __init__(self):
        self.save_dir = os.path.join(experiments_dir, model_name, model_params_subdir)
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)
        self.G = Generator(z_dim, hidden_dims_g, latent_size).cuda()
        self.D = Discriminator(latent_size, hidden_dims_d).cuda()

        # WGAN values from paper
        self.learning_rate_g = learning_rate_g
        self.learning_rate_d = learning_rate_d
        self.b1 = 0.5
        self.b2 = 0.999
        if batch_size == None:
            self.batch_size = 64
        else:
            self.batch_size = batch_size

        # WGAN_gradient penalty uses ADAM
        self.d_optimizer = optim.Adam(self.D.parameters(), lr=self.learning_rate_g, betas=(self.b1, self.b2))
        self.g_optimizer = optim.Adam(self.G.parameters(), lr=self.learning_rate_d, betas=(self.b1, self.b2))

        self.critic_iter = 5
        self.gp_lambda = gp_lambda


    def train_(self, generator_iters, train_loader):
        self.t_begin = t.time()

        # Now batches are callable self.data.next()
        self.data = self.get_infinite_batches(train_loader)

        one = torch.tensor(1, dtype=torch.float)
        mone = one * -1

        losses_d = []
        losses_g = []
        for g_iter in range(1, generator_iters + 1):
            # Train Discriminator
            for p in self.D.parameters():
                p.requires_grad = True

            d_loss_total = 0
            for d_iter in range(self.critic_iter):
                self.D.zero_grad()

                real_latent_codes = self.data.__next__().cuda()

                d_loss_real = self.D(real_latent_codes)
                d_loss_real = d_loss_real.mean()
                d_loss_real.backward(mone)

                z = torch.randn(self.batch_size, z_dim).cuda()

                fake_latent_codes = self.G(z)
                d_loss_fake = self.D(fake_latent_codes)
                d_loss_fake = d_loss_fake.mean()
                d_loss_fake.backward(one)

                gradient_penalty = self.calculate_gradient_penalty(real_latent_codes, fake_latent_codes)
                gradient_penalty.backward()

                d_loss = d_loss_fake - d_loss_real + gradient_penalty
                # Wasserstein_D = d_loss_real - d_loss_fake
                self.d_optimizer.step()
                # print(f'  Discriminator iteration: {d_iter + 1}/{self.critic_iter}, loss: {d_loss}')
                d_loss_total += d_loss.item()

            losses_d.append(d_loss_total)

            # Train Generator
            for p in self.D.parameters():
                p.requires_grad = False # to avoid update

            self.G.zero_grad()

            z = torch.randn(self.batch_size, z_dim).cuda()

            fake_latent_codes = self.G(z)
            g_loss = self.D(fake_latent_codes)
            g_loss = g_loss.mean()
            g_loss.backward(mone)

            self.g_optimizer.step()

            # print(f'Generator iteration: {g_iter}/{generator_iters}, g_loss: {g_loss}')
            losses_g.append(g_loss)

            print(f'Generator iteration: {g_iter}/{generator_iters}, g_loss: {g_loss}, d_loss: {d_loss_total}')

            if (g_iter) % SAVE_PER_ITERS == 0:
                self.save_model(g_iter)
                torch.save(
                    {
                        "loss_d": losses_d,
                        "loss_g": losses_g,
                    },
                    os.path.join(experiments_dir, model_name, "Logs.pth")
                )

        self.t_end = t.time()
        print('Time of training: {}'.format((self.t_end - self.t_begin)))

        # Save the trained parameters
        self.save_model("latest")

        torch.save(
            {
                "loss_d": losses_d,
                "loss_g": losses_g,
            },
            os.path.join(experiments_dir, model_name, "Logs.pth")
        )


    def train_yield(self, generator_iters, train_loader):
        self.t_begin = t.time()

        # Now batches are callable self.data.next()
        self.data = self.get_infinite_batches(train_loader)

        one = torch.tensor(1, dtype=torch.float)
        mone = one * -1

        losses_d = []
        losses_g = []
        for g_iter in range(1, generator_iters + 1):
            # Train Discriminator
            for p in self.D.parameters():
                p.requires_grad = True

            d_loss_total = 0
            for d_iter in range(self.critic_iter):
                self.D.zero_grad()

                real_latent_codes = self.data.__next__().cuda()

                d_loss_real = self.D(real_latent_codes)
                d_loss_real = d_loss_real.mean()
                d_loss_real.backward(mone)

                z = torch.randn(self.batch_size, z_dim).cuda()

                fake_latent_codes = self.G(z)
                d_loss_fake = self.D(fake_latent_codes)
                d_loss_fake = d_loss_fake.mean()
                d_loss_fake.backward(one)

                gradient_penalty = self.calculate_gradient_penalty(real_latent_codes, fake_latent_codes)
                gradient_penalty.backward()

                d_loss = d_loss_fake - d_loss_real + gradient_penalty
                # Wasserstein_D = d_loss_real - d_loss_fake
                self.d_optimizer.step()
                # print(f'  Discriminator iteration: {d_iter + 1}/{self.critic_iter}, loss: {d_loss}')
                d_loss_total += d_loss.item()

            losses_d.append(d_loss_total)

            # Train Generator
            for p in self.D.parameters():
                p.requires_grad = False # to avoid update

            self.G.zero_grad()

            z = torch.randn(self.batch_size, z_dim).cuda()

            fake_latent_codes = self.G(z)
            g_loss = self.D(fake_latent_codes)
            g_loss = g_loss.mean()
            g_loss.backward(mone)

            self.g_optimizer.step()

            # print(f'Generator iteration: {g_iter}/{generator_iters}, g_loss: {g_loss}')
            losses_g.append(g_loss)

            print(f'Generator iteration: {g_iter}/{generator_iters}, g_loss: {g_loss}, d_loss: {d_loss_total}')

            if (g_iter) % SAVE_PER_ITERS == 0:
                self.save_model(g_iter)
                torch.save(
                    {
                        "loss_d": losses_d,
                        "loss_g": losses_g,
                    },
                    os.path.join(experiments_dir, model_name, "Logs.pth")
                )

            yield

        self.t_end = t.time()
        print('Time of training: {}'.format((self.t_end - self.t_begin)))

        # Save the trained parameters
        self.save_model("latest")

        torch.save(
            {
                "loss_d": losses_d,
                "loss_g": losses_g,
            },
            os.path.join(experiments_dir, model_name, "Logs.pth")
        )


    def generate(self, num_codes=1, z=None):
        return self.G.generate(num_codes=num_codes, z=z)


    def calculate_gradient_penalty(self, real_latent_codes, fake_latent_codes):
        eta = torch.FloatTensor(self.batch_size,1).uniform_(0,1)
        eta = eta.expand(self.batch_size, real_latent_codes.size(1))
        eta = eta.cuda()

        interpolated = eta * real_latent_codes + ((1 - eta) * fake_latent_codes)

        interpolated = interpolated.cuda()

        # define it to calculate gradient
        interpolated = Variable(interpolated, requires_grad=True)

        # calculate probability of interpolated examples
        prob_interpolated = self.D(interpolated)

        # calculate gradients of probabilities with respect to examples
        gradients = autograd.grad(outputs=prob_interpolated, inputs=interpolated,
                grad_outputs=torch.ones(prob_interpolated.size()).cuda(),
                create_graph=True, retain_graph=True)[0]

        grad_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean() * self.gp_lambda
        return grad_penalty


    def save_model(self, iter):
        torch.save(self.G.state_dict(), os.path.join(self.save_dir, str(iter) + "_g.pth"))
        torch.save(self.D.state_dict(), os.path.join(self.save_dir, str(iter) + "_d.pth"))


    def load_model(self, load_dir, iter):
        filepath_d = os.path.join(load_dir, str(iter) + "_d.pth")
        filepath_g = os.path.join(load_dir, str(iter) + "_g.pth")
        self.D.load_state_dict(torch.load(filepath_d))
        self.G.load_state_dict(torch.load(filepath_g))
        print('Generator model loaded from {}.'.format(filepath_g))
        print('Discriminator model loaded from {}-'.format(filepath_d))


    def get_infinite_batches(self, data_loader):
        while True:
            for real_latent_codes in data_loader:
                yield real_latent_codes


    def eval(self):
        self.D.eval()
        self.G.eval()

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

