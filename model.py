import numpy as np
import os

import torch
import torch.nn as nn
import torch.nn.functional as F

from torchvision.utils import save_image, make_grid


from utils import *


class VAE(nn.Module):

    def __init__(self, device, batch_size, loader_train, loader_test, latent_dim,
                 low_size, middle_size, high_size, learning_rate, path, writer):
        super(VAE, self).__init__()

        self.device = device
        self.loader_train = loader_train
        self.loader_test = loader_test
        self.batch_size = batch_size

        self.latent_dim = latent_dim
        self.low_size = low_size
        self.middle_size = middle_size
        self.high_size = high_size
        #######################################################################
        #                       ** Fully Connected **
        #######################################################################
        self.fc1 = nn.Linear(self.high_size, self.middle_size)
        self.fc2 = nn.Linear(self.middle_size, self.low_size)
        self.fc31 = nn.Linear(self.low_size, self.latent_dim)
        self.fc32 = nn.Linear(self.low_size, self.latent_dim)
        self.fc4 = nn.Linear(self.latent_dim, self.low_size)
        self.fc5 = nn.Linear(self.low_size, self.middle_size)
        self.fc6 = nn.Linear(self.middle_size, self.high_size)

        self.optimizer = torch.optim.Adam(self.parameters(), lr=learning_rate)
        self.to(device)
        self.path = path

        self.images, _ = next(iter(self.loader_train))
        self.grid = make_grid(self.images)
        self.writer = writer

        self.writer.add_image('images', self.grid, 0)
        self.writer.add_graph(self, self.images)

    def encode(self, x):
        #######################################################################
        #                       ** ReLU Activation **
        #######################################################################
        h1 = F.relu(self.fc1(x))
        h2 = F.relu(self.fc2(h1))
        return self.fc31(h2), self.fc32(h2)

    @staticmethod
    def reparametrize(mu, logvar):
        #######################################################################
        #                       ** Reparametrization trick **
        #######################################################################
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        #######################################################################
        #                       ** ReLU Activation **
        #######################################################################
        h4 = F.relu(self.fc4(z))
        h5 = F.relu(self.fc5(h4))
        return torch.sigmoid(self.fc6(h5))

    def forward(self, x):
        mu, logvar = self.encode(x.view(-1, self.high_size))
        z = self.reparametrize(mu, logvar)
        return self.decode(z), mu, logvar

    @staticmethod
    def loss_function(recon_x, x, mu, logvar):

        # Kullback_Leibler Divergence
        KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

        # Binary Cross Entropy is used as an alternative to Negative Log Likelihood
        NLL = F.binary_cross_entropy(recon_x, x.view(-1, 784), reduction='sum')

        return [KLD, NLL]

    def summary(self):
        params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print("Total number of parameters is: {}".format(params))
        print(self)

    def test(self, epoch_index):

        self.eval()
        KLD_loss, NLL_loss = [0, 0]

        with torch.no_grad():
            for i, (data, _) in enumerate(self.loader_test):
                data = data.to(self.device)
                recon_batch, mu, logvar = self(data)
                loss_2d = self.loss_function(recon_batch, data, mu, logvar)
                KLD_loss += loss_2d[0].item()
                NLL_loss += loss_2d[1].item()
                if i == 0:
                    n = min(data.size(0), 8)
                    comparison = torch.cat([data[:n], recon_batch.view(self.batch_size, 1, 28, 28)[:n]])
                    if not os.path.exists(self.path + '/reconstructions/'):
                        os.makedirs(self.path + '/reconstructions/')
                    save_image(comparison.cpu(), self.path + '/reconstructions/reconstructions_epoch_'
                               + str(epoch_index) + '.png', nrow=n)

        KLD_loss /= len(self.loader_test.dataset)
        NLL_loss /= len(self.loader_test.dataset)
        return KLD_loss, NLL_loss

    def perform_training(self, num_epochs):
        self.train()
        total_loss_per_epoch_training = []
        total_loss_per_epoch_testing = []

        for epoch in range(num_epochs):

            KLD_loss, NLL_loss = [0, 0]

            for batch_idx, (data, _) in enumerate(self.loader_train):
                data = data.to(self.device)
                self.optimizer.zero_grad()
                recon_batch, mu, logvar = self(data)
                loss_2d = self.loss_function(recon_batch, data, mu, logvar)
                loss = sum(loss_2d)
                loss.backward()
                KLD_loss += loss_2d[0].item()
                NLL_loss += loss_2d[1].item()
                self.optimizer.step()
            kld_total_loss, nll_total_loss = [KLD_loss / len(self.loader_train.dataset),
                                              NLL_loss / len(self.loader_train.dataset)]

            self.writer.add_scalar('kld loss', kld_total_loss, epoch)
            self.writer.add_scalar('nll loss', nll_total_loss, epoch)

            total_loss_per_epoch_training.append([kld_total_loss, nll_total_loss])

            total_loss_per_epoch_testing.append(self.test(epoch))

            print(f"====> Epoch: {epoch} Average loss: {(KLD_loss + NLL_loss) / len(self.loader_train.dataset)}")

        return np.array(total_loss_per_epoch_training), np.array(total_loss_per_epoch_testing)

    def save(self, filename):
        torch.save(self.state_dict(), filename)


class Generator(nn.Module):
    def __init__(self, latent_vector_size, nb_filter, learning_rate, beta1=0.5):
        super(Generator, self).__init__()

        self.model = nn.Sequential(nn.Linear(latent_vector_size, 4 * 4 * 256), nn.LeakyReLU())

        self.cnn = nn.Sequential(
            nn.ConvTranspose2d(nb_filter * 8, nb_filter * 4, 3, stride=2, padding=0, output_padding=0),
            nn.LeakyReLU(),
            nn.ConvTranspose2d(nb_filter * 4, nb_filter * 2, 3, stride=2, padding=1, output_padding=0),
            nn.LeakyReLU(),
            nn.ConvTranspose2d(nb_filter * 2, nb_filter * 2, 3, stride=2, padding=2, output_padding=1),
            nn.LeakyReLU(),
            nn.Conv2d(nb_filter * 2, 3, 3, stride=1, padding=1),
            nn.Tanh()
        )

        self.optimizer = torch.optim.Adam(self.parameters(), lr=learning_rate, betas=(beta1, 0.999))

    def decode(self, z):
        x = self.model(z)
        x = x.view(-1, 256, 4, 4)
        x = self.cnn(x)
        return x

    def forward(self, z):
        return self.decode(z)

    def summary(self):
        params_G = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print("Total number of parameters in Generator is: {}".format(params_G))
        print(self)
        print('\n')


class Discriminator(nn.Module):
    def __init__(self, nb_filter, learning_rate, beta1=0.5):
        super(Discriminator, self).__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(3, nb_filter * 2, 3, stride=1, padding=1),
            nn.LeakyReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(64, 128, 3, stride=1, padding=1),
            nn.LeakyReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(128, 256, 3, stride=1, padding=1),
            nn.LeakyReLU(),
            nn.MaxPool2d(2, 2)
        )
        self.fc = nn.Sequential(
            nn.Linear(4 * 4 * 256, 128),
            nn.LeakyReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )

        self.optimizer = torch.optim.Adam(self.parameters(), lr=learning_rate, betas=(beta1, 0.999))

    def discriminator(self, x):
        out = self.cnn(x)
        out = out.view(-1, 4 * 4 * 256)
        out = self.fc(out)
        return out

    def forward(self, x):
        out = self.discriminator(x)
        return out.view(-1, 1).squeeze(1)

    def summary(self):
        params_D = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print("Total number of parameters in Discriminator is: {}".format(params_D))
        print(self)
        print('\n')


class DCGAN():
    def __init__(self, device, batch_size, loader_train, loader_val, loader_testing,
                 latent_vector_size, nb_filter, learning_rate, path, writer, use_weights_init=True):

        self.generator = Generator(latent_vector_size, nb_filter, learning_rate).to(device)
        self.discriminator = Discriminator(nb_filter, learning_rate).to(device)

        self.loader_train = loader_train
        self.loader_val = loader_val
        self.loader_testing = loader_testing

        if use_weights_init:
            self.weights_init()

        self.fixed_noise = torch.randn(batch_size, latent_vector_size, device=device)
        self.real_label = 1
        self.fake_label = 0

        self.batch_size = batch_size
        self.latent_vector_size = latent_vector_size
        self.device = device
        self.path = path

        self.images, _ = next(iter(self.loader_train))
        self.grid = make_grid(self.images)
        self.writer = writer

        self.writer.add_image('images', self.grid, 0)
        # self.writer.add_graph(self.generator)
        # self.writer.add_graph(self.discriminator)

    def weights_init(self):
        pass

    def summary(self):
        self.generator.summary()
        self.discriminator.summary()

    @staticmethod
    def loss_function(out, label):
        criterion = nn.BCELoss(reduction='mean')
        loss = criterion(out, label)
        return loss

    def perform_training(self, num_epochs):

        train_losses_G = []
        train_losses_D = []

        for epoch in range(num_epochs):
            for i, data in enumerate(self.loader_train, 0):
                train_loss_G, train_loss_D = [0, 0]

                #######################################################################
                #                       ** UPDATE DISCRIMINATOR **
                #              ** maximize log(D(x)) + log(1 - D(G(z))) **
                #######################################################################

                # train with real
                self.discriminator.zero_grad()
                real_cpu = data[0].to(self.device)
                batch_size = real_cpu.size(0)
                label = torch.ones(batch_size, device=self.device)
                output = self.discriminator.forward(real_cpu)
                errD_real = self.loss_function(output, label)
                errD_real.backward()
                D_x = output.mean().item()

                # train with fake
                noise = torch.randn(batch_size, self.latent_vector_size, device=self.device)
                fake = self.generator(noise)
                label.fill_(self.fake_label)
                output = self.discriminator(fake.detach())

                errD_fake = self.loss_function(output, label)
                errD_fake.backward()
                D_G_z1 = output.mean().item()
                errD = errD_real + errD_fake
                train_loss_D += errD.item()
                self.discriminator.optimizer.step()

                #######################################################################
                #                       ** UPDATE GENERATOR **
                #                     ** maximize log(D(G(z))) **
                #######################################################################

                self.generator.zero_grad()
                label.fill_(self.real_label)  # fake labels are real for generator cost
                output = self.discriminator(fake)
                errG = self.loss_function(output, label)
                errG.backward()
                D_G_z2 = output.mean().item()
                train_loss_G += errG.item()
                self.generator.optimizer.step()

                print('[%d/%d][%d/%d] Loss_D: %.4f Loss_G: %.4f D(x): %.4f D(G(z)): %.4f / %.4f'
                      % (epoch, num_epochs, i, len(self.loader_train),
                         errD.item(), errG.item(), D_x, D_G_z1, D_G_z2))

                if epoch == 0:
                    save_image(denorm(real_cpu.cpu()).float(), self.path + '/real_samples.png')

                fake = self.generator(self.fixed_noise)
                save_image(denorm(fake.cpu()).float(), self.path + '/fake_samples_epoch_%03d.png' % epoch)

                discriminator_loss = train_loss_D / len(self.loader_train)
                generator_loss = train_loss_G / len(self.loader_train)
                train_losses_D.append(discriminator_loss)
                train_losses_G.append(generator_loss)

                self.writer.add_scalar('discriminator loss', discriminator_loss, epoch)
                self.writer.add_scalar('generator loss', generator_loss, epoch)

        return np.array(train_losses_G), np.array(train_losses_D)

    def save(self):

        # save losses and models
        torch.save(self.generator.state_dict(), self.path + '/DCGAN_model_G.pth')
        torch.save(self.discriminator.state_dict(), self.path + '/DCGAN_model_D.pth')
