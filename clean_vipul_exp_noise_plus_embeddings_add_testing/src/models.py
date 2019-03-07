"""
Copyright (C) 2017 NVIDIA Corporation.  All rights reserved.
Licensed under the CC BY-NC-ND 4.0 license (https://creativecommons.org/licenses/by-nc-nd/4.0/legalcode).
"""

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.utils.data
from torch.autograd import Variable
from torch.nn import functional as F

import numpy as np

if torch.cuda.is_available():
    T = torch.cuda
else:
    T = torch


class Noise(nn.Module):
    def __init__(self, use_noise, sigma=0.2):
        super(Noise, self).__init__()
        self.use_noise = use_noise
        self.sigma = sigma

    def forward(self, x):
        if self.use_noise:
            return x + self.sigma * Variable(T.FloatTensor(x.size()).normal_(), requires_grad=False)
        return x

class View(nn.Module):
    def __init__(self):
        super(View, self).__init__()

    def forward(self, x):
        return x.view(-1,64)


class ImageDiscriminator(nn.Module):
    def __init__(self, n_channels, ndf=64, use_noise=False, noise_sigma=None):
        super(ImageDiscriminator, self).__init__()

        self.use_noise = use_noise

        self.main = nn.Sequential(
            Noise(use_noise, sigma=noise_sigma),
            nn.Conv2d(n_channels, ndf, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),

            Noise(use_noise, sigma=noise_sigma),
            nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),

            Noise(use_noise, sigma=noise_sigma),
            nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),

            Noise(use_noise, sigma=noise_sigma),
            nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(ndf * 8, 1, 4, 1, 0, bias=False),
        )

    def forward(self, input):
        h = self.main(input).squeeze()
        return h, None


class PatchImageDiscriminator(nn.Module):
    def __init__(self, n_channels, ndf=64, use_noise=False, noise_sigma=None):
        super(PatchImageDiscriminator, self).__init__()

        self.use_noise = use_noise

        self.main = nn.Sequential(
            Noise(use_noise, sigma=noise_sigma),
            nn.Conv2d(n_channels, ndf, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),

            Noise(use_noise, sigma=noise_sigma),
            nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),

            Noise(use_noise, sigma=noise_sigma),
            nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),

            Noise(use_noise, sigma=noise_sigma),
            nn.Conv2d(ndf * 4, 1, 4, 2, 1, bias=False),
        )

    def forward(self, input):
        h = self.main(input).squeeze()
        return h, None


class PatchVideoDiscriminator(nn.Module):
    def __init__(self, n_channels, n_output_neurons=1, bn_use_gamma=True, use_noise=False, noise_sigma=None, ndf=64):
        super(PatchVideoDiscriminator, self).__init__()

        self.n_channels = n_channels
        self.n_output_neurons = n_output_neurons
        self.use_noise = use_noise
        self.bn_use_gamma = bn_use_gamma

        self.main = nn.Sequential(
            Noise(use_noise, sigma=noise_sigma),
            nn.Conv3d(n_channels, ndf, 4, stride=(1, 2, 2), padding=(0, 1, 1), bias=False),
            nn.LeakyReLU(0.2, inplace=True),

            Noise(use_noise, sigma=noise_sigma),
            nn.Conv3d(ndf, ndf * 2, 4, stride=(1, 2, 2), padding=(0, 1, 1), bias=False),
            nn.BatchNorm3d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),

            Noise(use_noise, sigma=noise_sigma),
            nn.Conv3d(ndf * 2, ndf * 4, 4, stride=(1, 2, 2), padding=(0, 1, 1), bias=False),
            nn.BatchNorm3d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv3d(ndf * 4, 1, 4, stride=(1, 2, 2), padding=(0, 1, 1), bias=False),
        )

    def forward(self, input):
        h = self.main(input).squeeze()

        return h, None


class VideoDiscriminator(nn.Module):
    def __init__(self, n_channels, n_output_neurons=1, bn_use_gamma=True, use_noise=False, noise_sigma=None, ndf=64):
        super(VideoDiscriminator, self).__init__()

        self.n_channels = n_channels
        self.n_output_neurons = n_output_neurons
        self.use_noise = use_noise
        self.bn_use_gamma = bn_use_gamma

        self.main = nn.Sequential(
            Noise(use_noise, sigma=noise_sigma),
            nn.Conv3d(n_channels, ndf, 4, stride=(1, 2, 2), padding=(0, 1, 1), bias=False),
            nn.LeakyReLU(0.2, inplace=True),

            Noise(use_noise, sigma=noise_sigma),
            nn.Conv3d(ndf, ndf * 2, 4, stride=(1, 2, 2), padding=(0, 1, 1), bias=False),
            nn.BatchNorm3d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),

            Noise(use_noise, sigma=noise_sigma),
            nn.Conv3d(ndf * 2, ndf * 4, 4, stride=(1, 2, 2), padding=(0, 1, 1), bias=False),
            nn.BatchNorm3d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),

            Noise(use_noise, sigma=noise_sigma),
            nn.Conv3d(ndf * 4, ndf * 8, 4, stride=(1, 2, 2), padding=(0, 1, 1), bias=False),
            nn.BatchNorm3d(ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv3d(ndf * 8, n_output_neurons, 4, 1, 0, bias=False),
        )

    def forward(self, input):
        h = self.main(input).squeeze()

        return h, None


class CategoricalVideoDiscriminator(VideoDiscriminator):
    def __init__(self, n_channels, dim_categorical, n_output_neurons=1, use_noise=False, noise_sigma=None):
        super(CategoricalVideoDiscriminator, self).__init__(n_channels=n_channels,
                                                            n_output_neurons=n_output_neurons + dim_categorical,
                                                            use_noise=use_noise,
                                                            noise_sigma=noise_sigma)

        self.dim_categorical = dim_categorical

    def split(self, input):
        return input[:, :input.size(1) - self.dim_categorical], input[:, input.size(1) - self.dim_categorical:]

    def forward(self, input):
        h, _ = super(CategoricalVideoDiscriminator, self).forward(input)
        labels, categ = self.split(h)
        return labels, categ


class VideoGenerator(nn.Module):
    def __init__(self, n_channels, dim_z_content, dim_z_category, dim_z_motion,
                 video_length, ngf=64):
        super(VideoGenerator, self).__init__()

        self.n_channels = n_channels
        self.dim_z_content = dim_z_content
        self.dim_z_category = dim_z_category
        self.dim_z_motion = dim_z_motion
        self.video_length = video_length

        dim_z = dim_z_motion + dim_z_category + dim_z_content+64

        self.recurrent = nn.GRUCell(dim_z_motion, dim_z_motion)

        self.main = nn.Sequential(
            nn.ConvTranspose2d(dim_z, ngf * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(ngf * 8),
            nn.ReLU(True),
            nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True),
            nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True),
            nn.ConvTranspose2d(ngf * 2, ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),
            nn.ConvTranspose2d(ngf, self.n_channels, 4, 2, 1, bias=False),
            nn.Tanh()
        )


        self.main2 = nn.Sequential(
            nn.Conv2d(n_channels, 64, 4, 2, 1, bias=False),
            nn.ReLU(True),

            nn.Conv2d(64, 64 * 2, 4, 2, 1, bias=False),
            nn.ReLU(True),

            nn.Conv2d(64 * 2, 1, 4, 2, 1, bias=False),
            nn.ReLU(True),
            View(),
            nn.Linear(64, 64),
            nn.ReLU(True),
            nn.Linear(64, 64),
            nn.ReLU(True)

        )

        # self.main2 = nn.Sequential(
        #     nn.Conv2d(n_channels, 64, 3, 2, bias=False),
        #     nn.ReLU(True),
        #     nn.Conv2d(64, 64, 3, 1, bias=False),
        #     nn.ReLU(True),
        #     nn.MaxPool2d(64, 64, 2, 2),


        #     # nn.Conv2d(64, 128, 3, 1, bias=False),
        #     # nn.ReLU(True),
        #     # nn.Conv2d(128, 128, 3, 1, bias=False),
        #     # nn.ReLU(True),
        #     # nn.MaxPool2d(128, 128, 2, 2),


        #     # nn.Conv2d(128, 256, 3, 1, bias=False),
        #     # nn.ReLU(True),
        #     # nn.Conv2d(256, 256, 3, 1, bias=False),
        #     # nn.ReLU(True),
        #     # nn.MaxPool2d(256, 256, 2, 2),

        #     # nn.Conv2d(256, 512, 3, 1, bias=False),
        #     # nn.ReLU(True),
        #     # nn.Conv2d(512, 512, 3, 1, bias=False),
        #     # nn.ReLU(True),
        #     # nn.MaxPool2d(512, 512, 14, 14),

        #     # nn.Linear(512, 128),
        #     # nn.ReLU(True),
        #     # nn.Linear(128, 64),
        #     # nn.ReLU(True)
        # # nn.Linear(6, latent_dim)

        # # nn.Conv2d(ndf * 4, 1, 4, 2, 1, bias=False),
        # )





    def sample_z_m(self, num_samples, video_len=None):
        video_len = video_len if video_len is not None else self.video_length

        h_t = [self.get_gru_initial_state(num_samples)]

        for frame_num in range(video_len):
            e_t = self.get_iteration_noise(num_samples)
            h_t.append(self.recurrent(e_t, h_t[-1]))

        z_m_t = [h_k.view(-1, 1, self.dim_z_motion) for h_k in h_t]
        z_m = torch.cat(z_m_t[1:], dim=1).view(-1, self.dim_z_motion)

        return z_m

    def sample_z_categ(self, num_samples, video_len):
        video_len = video_len if video_len is not None else self.video_length

        classes_to_generate = np.random.randint(self.dim_z_category, size=num_samples)
        one_hot = np.zeros((num_samples, self.dim_z_category), dtype=np.float32)
        one_hot[np.arange(num_samples), classes_to_generate] = 1
        one_hot_video = np.repeat(one_hot, video_len, axis=0)

        one_hot_video = torch.from_numpy(one_hot_video)

        if torch.cuda.is_available():
            one_hot_video = one_hot_video.cuda()

        return Variable(one_hot_video), classes_to_generate

    def sample_z_content(self, num_samples, video_len=None):
        video_len = video_len if video_len is not None else self.video_length

        content = np.random.normal(0, 1, (num_samples, self.dim_z_content)).astype(np.float32)
        content = np.repeat(content, video_len, axis=0)
        content = torch.from_numpy(content)
        if torch.cuda.is_available():
            content = content.cuda()
        return Variable(content)

    def sample_z_video(self, num_samples, video_len=None):
        z_content = self.sample_z_content(num_samples, video_len)
        z_category, z_category_labels = self.sample_z_categ(num_samples, video_len)
        z_motion = self.sample_z_m(num_samples, video_len)

        z = torch.cat([z_content, z_category, z_motion], dim=1)

        return z, z_category_labels

    def sample_videos(self, num_samples,input_batch, video_len=None):
        video_len = video_len if video_len is not None else self.video_length

        input_batch = input_batch.permute(0, 2, 1, 3, 4)
        input_batch=input_batch.contiguous()
        input_batch=input_batch.view(input_batch.size(0)*input_batch.size(1),input_batch.size(2),input_batch.size(3),input_batch.size(4))
        latent=self.main2(input_batch)
        latent=latent.view(latent.size(0), latent.size(1), 1, 1)

        # audio_cond=audio_cond.view(audio_cond.size(0), audio_cond.size(1), 1, 1)


        z, z_category_labels = self.sample_z_video(num_samples, video_len)
        z=z.view(z.size(0), z.size(1), 1, 1)
        # print('z video_dim',z.shape)
        z=torch.cat((z, latent), 1)
        # z=torch.cat((z, audio_cond), 1)
        h = self.main(z)
        h = h.view(h.size(0) / video_len, video_len, self.n_channels, h.size(3), h.size(3))

        z_category_labels = torch.from_numpy(z_category_labels)

        if torch.cuda.is_available():
            z_category_labels = z_category_labels.cuda()

        h = h.permute(0, 2, 1, 3, 4)
        return h, latent.view(latent.size(0), latent.size(1)), Variable(z_category_labels, requires_grad=False)

    def sample_images(self, num_samples,input_batch):
        z, z_category_labels = self.sample_z_video(num_samples * self.video_length * 2)

        j = np.sort(np.random.choice(z.size(0), num_samples, replace=False)).astype(np.int64)
        z = z[j, ::]
        z = z.view(z.size(0), z.size(1), 1, 1)
        # print('z image dim',z.shape)


        latent=self.main2(input_batch)
        # print('latent',latent.shape)
        latent=latent.view(latent.size(0), latent.size(1), 1, 1)
        # audio_cond=audio_cond.view(audio_cond.size(0), audio_cond.size(1), 1, 1)
        z=torch.cat((z, latent), 1)
        # z=torch.cat((z, audio_cond), 1)
        h = self.main(z)

        return h,latent.view(latent.size(0), latent.size(1)), None

    def get_gru_initial_state(self, num_samples):
        return Variable(T.FloatTensor(num_samples, self.dim_z_motion).normal_())

    def get_iteration_noise(self, num_samples):
        return Variable(T.FloatTensor(num_samples, self.dim_z_motion).normal_())


class ImageConvNet(nn.Module):

    def __init__(self):
        super(ImageConvNet, self).__init__()
        self.pool = nn.MaxPool2d(2, stride=2)

        self.cnn1 = nn.Conv2d(3, 64, 3, stride=2, padding=1)
        self.cnn2 = nn.Conv2d(64, 64, 3, padding=1)
        self.bat10 = nn.BatchNorm2d(64)
        self.bat11 = nn.BatchNorm2d(64)

        self.cnn3 = nn.Conv2d(64, 128, 3, stride=1, padding=1)
        self.cnn4 = nn.Conv2d(128, 128, 3, padding=1)
        self.bat20 = nn.BatchNorm2d(128)
        self.bat21 = nn.BatchNorm2d(128)

        self.cnn5 = nn.Conv2d(128, 256, 3, stride=1, padding=1)
        self.cnn6 = nn.Conv2d(256, 256, 3, padding=1)
        self.bat30 = nn.BatchNorm2d(256)
        self.bat31 = nn.BatchNorm2d(256)

        self.cnn7 = nn.Conv2d(256, 512, 3, stride=1, padding=1)
        self.cnn8 = nn.Conv2d(512, 512, 3, padding=1)
        self.bat40 = nn.BatchNorm2d(512)
        self.bat41 = nn.BatchNorm2d(512)

        self.vpool4  = nn.MaxPool2d(14, stride=14)
        self.vfc1    = nn.Linear(512, 512)
        self.vl2norm = nn.BatchNorm1d(512)


    def forward(self, inp):
        c = F.relu(self.bat10(self.cnn1(inp)))
        c = F.relu(self.bat11(self.cnn2(c)))
        c = self.pool(c)

        c = F.relu(self.bat20(self.cnn3(c)))
        c = F.relu(self.bat21(self.cnn4(c)))
        c = self.pool(c)

        c = F.relu(self.bat30(self.cnn5(c)))
        c = F.relu(self.bat31(self.cnn6(c)))
        c = self.pool(c)

        c = F.relu(self.bat40(self.cnn7(c)))
        c = F.relu(self.bat41(self.cnn8(c)))

    #Bx512x14x14
        # print('SIZE OF c1',c.shape)
        c = self.vpool4(c).squeeze(2).squeeze(2)
        # print('SIZE OF c2',c.shape)
        c = F.relu(self.vfc1(c))
        c = self.vl2norm(c)
        return c

    # Dummy function, just to check if feedforward is working or not
    def loss(self, output):
        return (output.mean())**2


# ngpu = 1

class ImageGeneratorDC(nn.Module):

    def __init__(self, nz = 512, ngf = 64 , nc=3 ):
        super(ImageGeneratorDC, self).__init__()
        # self.ngpu = ngpu
        self.main = nn.Sequential(
            # input is Z, going into a convolution
            nn.ConvTranspose2d( nz, ngf * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(ngf * 8),
            nn.ReLU(True),
            # state size. (ngf*8) x 4 x 4
            nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True),
            # state size. (ngf*4) x 8 x 8
            nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True),
            # state size. (ngf*2) x 16 x 16
            nn.ConvTranspose2d(ngf * 2,     ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),
            # state size. (ngf) x 32 x 32
            nn.ConvTranspose2d(    ngf,      nc, 4, 2, 1, bias=False),
            nn.Tanh()
            # state size. (nc) x 64 x 64
        )

    def forward(self, input):


        output = self.main(input)
        return output


class ImageDiscriminatorDC(nn.Module):

    def __init__(self,  nz = 512 , nc=3 , ndf=64):
        super(ImageDiscriminatorDC, self).__init__()
        # self.ngpu = ngpu
        self.main = nn.Sequential(
            # input is (nc) x 64 x 64
            nn.Conv2d(nc, ndf, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf) x 32 x 32
            nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*2) x 16 x 16
            nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*4) x 8 x 8
            nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*8) x 4 x 4
            nn.Conv2d(ndf * 8, 1, 4, 1, 0, bias=False),
            nn.Sigmoid()
        )

    def forward(self, input):

        output = self.main(input)

        return output.view(-1, 1).squeeze(1)



class ImageEncoder64(nn.Module):
    def __init__(self, n_channels, ndf=64, use_noise=False, noise_sigma=None):
        super(PatchImageDiscriminator, self).__init__()

        self.use_noise = use_noise

        self.main = nn.Sequential(
            nn.Conv2d(n_channels, ndf, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),

            # nn.Conv2d(ndf * 4, 1, 4, 2, 1, bias=False),
        )

    def forward(self, input):
        h = self.main(input).squeeze()
        return h, None
