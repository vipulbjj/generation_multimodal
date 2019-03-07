"""
Copyright (C) 2017 NVIDIA Corporation.  All rights reserved.
Licensed under the CC BY-NC-ND 4.0 license (https://creativecommons.org/licenses/by-nc-nd/4.0/legalcode).
"""

import os
import time

import numpy as np

from logger import Logger

import torch
from torch import nn
import gc

from torch.autograd import Variable
import torch.optim as optim

if torch.cuda.is_available():
    T = torch.cuda
else:
    T = torch

#for 224 to 64
from torchvision import transforms
from PIL import Image


##audio
from utils import *  ##for time slice, numpy_to_var,calc_gradient_penalty
from torch import autograd
from utils import save_samples


import warnings
def fxn():
    warnings.warn("deprecated", DeprecationWarning)

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    fxn()



def images_to_numpy(tensor):
    generated = tensor.data.cpu().numpy().transpose(0, 2, 3, 1)
    generated[generated < -1] = -1
    generated[generated > 1] = 1
    generated = (generated + 1) / 2 * 255
    return generated.astype('uint8')


def videos_to_numpy(tensor):
    generated = tensor.data.cpu().numpy().transpose(0, 1, 2, 3, 4)
    generated[generated < -1] = -1
    generated[generated > 1] = 1
    generated = (generated + 1) / 2 * 255
    return generated.astype('uint8')


def one_hot_to_class(tensor):
    a, b = np.nonzero(tensor)
    return np.unique(b).astype(np.int32)


class Trainer(object):
    def __init__(self, image_sampler,image_sampler_test, log_interval, train_batches, log_folder,LOGGER_audio,LOGGER_image, use_cuda=False,
                 use_infogan=True, use_categories=True):

        self.use_categories = use_categories

        self.gan_criterion = nn.BCEWithLogitsLoss()
        self.category_criterion = nn.CrossEntropyLoss()

        self.image_sampler = image_sampler
        self.image_sampler_test=image_sampler_test

        self.image_batch_size = self.image_sampler.batch_size
        self.image_batch_size_test = self.image_sampler_test.batch_size

        self.audio_batch_size = self.image_sampler.batch_size
        self.audio_batch_size_test = self.image_sampler_test.batch_size


        self.log_interval = log_interval
        self.train_batches = train_batches

        self.log_folder = log_folder

        self.use_cuda = use_cuda
        self.use_infogan = use_infogan

        self.image_enumerator = None

        #audio part
        self.LOGGER_audio=LOGGER_audio

        self.LOGGER_image=LOGGER_image

    @staticmethod
    def ones_like(tensor, val=1.):
        return Variable(T.FloatTensor(tensor.size()).fill_(val), requires_grad=False)

    @staticmethod
    def zeros_like(tensor, val=0.):
        return Variable(T.FloatTensor(tensor.size()).fill_(val), requires_grad=False)

    # def compute_gan_loss(self, discriminator, sample_true, sample_fake, is_video):
    #     real_batch = sample_true()

    #     batch_size = real_batch['images'].size(0)
    #     fake_batch, generated_categories = sample_fake(batch_size)

    #     real_labels, real_categorical = discriminator(Variable(real_batch['images']))
    #     fake_labels, fake_categorical = discriminator(fake_batch)

    #     fake_gt, real_gt = self.get_gt_for_discriminator(batch_size, real=0.)

    #     l_discriminator = self.gan_criterion(real_labels, real_gt) + \
    #                       self.gan_criterion(fake_labels, fake_gt)

    #     # update image discriminator here

    #     # sample again for videos

    #     # update video discriminator

    #     # sample again
    #     # - videos
    #     # - images

    #     # l_vidoes + l_images -> l
    #     # l.backward()
    #     # opt.step()


    #     #  sample again and compute for generator

    #     fake_gt = self.get_gt_for_generator(batch_size)
    #     # to real_gt
    #     l_generator = self.gan_criterion(fake_labels, fake_gt)

    #     if is_video:

    #         # Ask the video discriminator to learn categories from training videos
    #         categories_gt = Variable(torch.squeeze(real_batch['categories'].long()))
    #         l_discriminator += self.category_criterion(real_categorical, categories_gt)

    #         if self.use_infogan:
    #             # Ask the generator to generate categories recognizable by the discriminator
    #             l_generator += self.category_criterion(fake_categorical, generated_categories)

    #     return l_generator, l_discriminator

    def sample_real_image_batch(self):
        if self.image_enumerator is None:
            self.image_enumerator = enumerate(self.image_sampler)

        batch_idx, batch = next(self.image_enumerator)
        b = batch
        if self.use_cuda:
            for k, v in batch.iteritems():
                b[k] = v.cuda()

        if batch_idx == len(self.image_sampler) - 1:
            self.image_enumerator = enumerate(self.image_sampler)

        return b


    def sample_test_image_batch(self):
        if self.image_enumerator is None:
            self.image_enumerator = enumerate(self.image_sampler_test)

        batch_idx, batch = next(self.image_enumerator)
        b = batch
        if self.use_cuda:
            for k, v in batch.iteritems():
                b[k] = v.cuda()

        if batch_idx == len(self.image_sampler_test) - 1:
            self.image_enumerator = enumerate(self.image_sampler_test)

        return b

    def sample_real_video_batch(self):
        if self.video_enumerator is None:
            self.video_enumerator = enumerate(self.video_sampler)

        batch_idx, batch = next(self.video_enumerator)
        b = batch
        if self.use_cuda:
            for k, v in batch.iteritems():
                b[k] = v.cuda()

        if batch_idx == len(self.video_sampler) - 1:
            self.video_enumerator = enumerate(self.video_sampler)

        return b

    # def train_discriminator(self, discriminator, real_batch, sample_fake, opt, batch_size, use_categories):
    #     opt.zero_grad()

    #     # real_batch = sample_true()
    #     batch = Variable(real_batch['images'], requires_grad=False)

    #     # util.show_batch(batch.data)

    #     fake_batch,image_cond, generated_categories = sample_fake(batch_size,batch)

    #     real_labels, real_categorical = discriminator(batch)
    #     fake_labels, fake_categorical = discriminator(fake_batch.detach())

    #     ones = self.ones_like(real_labels)
    #     zeros = self.zeros_like(fake_labels)

    #     l_discriminator = self.gan_criterion(real_labels, ones) + \
    #                       self.gan_criterion(fake_labels, zeros)

    #     if use_categories:
    #         # Ask the video discriminator to learn categories from training videos
    #         categories_gt = Variable(torch.squeeze(real_batch['categories'].long()), requires_grad=False)
    #         l_discriminator += self.category_criterion(real_categorical.squeeze(), categories_gt)

    #     l_discriminator.backward()
    #     opt.step()

    #     return l_discriminator,image_cond

    # def train_generator(self,
    #                     image_discriminator, video_discriminator,
    #                     sample_fake_images,real_batch, sample_fake_videos,real_batch_video,
    #                     opt):

    #     opt.zero_grad()

    #     # train on images
    #     batch_image = Variable(real_batch['images'], requires_grad=False)
    #     fake_batch,image_cond, generated_categories = sample_fake_images(self.image_batch_size,batch_image)
    #     fake_labels, fake_categorical = image_discriminator(fake_batch)
    #     all_ones = self.ones_like(fake_labels)

    #     l_generator = self.gan_criterion(fake_labels, all_ones)

    #     # train on videos
    #     batch_video = Variable(real_batch_video['images'], requires_grad=False)
    #     fake_batch,video_cond, generated_categories = sample_fake_videos(self.video_batch_size,batch_video)
    #     fake_labels, fake_categorical = video_discriminator(fake_batch)
    #     all_ones = self.ones_like(fake_labels)

    #     l_generator += self.gan_criterion(fake_labels, all_ones)

    #     if self.use_infogan:
    #         # Ask the generator to generate categories recognizable by the discriminator
    #         l_generator += self.category_criterion(fake_categorical.squeeze(), generated_categories)

    #     l_generator.backward(retain_graph=True)
    #     opt.step()

    #     return l_generator,image_cond,video_cond

    def train(self,ImageModel,netG,netD,audio_encoder,ImageGeneratorModel,ImageDiscriminatorModel,args):
        if self.use_cuda:
            netG.cuda()
            netD.cuda()
            ImageModel.cuda()
            audio_encoder.cuda()
            ImageGeneratorModel.cuda()
            ImageDiscriminatorModel.cuda()

        logger = Logger(self.log_folder)
        ##audio part
        optimizerG = optim.Adam(netG.parameters(), lr=float(args['--learning_rate']), betas=(float(args['--beta1']), float(args['--beta2'])))
        optimizerD = optim.Adam(netD.parameters(), lr=float(args['--learning_rate']), betas=(float(args['--beta1']), float(args['--beta2'])))
        opt_generator = optim.Adam(ImageModel.parameters(), lr=0.0002, betas=(0.5, 0.999), weight_decay=0.00001)


        optimizerG_Image = optim.Adam(ImageGeneratorModel.parameters(), lr=float(args['--learning_rate']), betas=(float(args['--beta1']), float(args['--beta2'])))
        optimizerD_image = optim.Adam(ImageDiscriminatorModel.parameters(), lr=float(args['--learning_rate']), betas=(float(args['--beta1']), float(args['--beta2'])))
        opt_encoder = optim.Adam(audio_encoder.parameters(), lr=float(args['--learning_rate']), betas=(float(args['--beta1']), float(args['--beta2'])))

        # opt_encoder = optim.Adam(audio_encoder.parameters(), lr=0.0002, betas=(0.5, 0.999), weight_decay=0.00001)
        # =============Train===============
        lmbda = float(args['--lmbda'])
        epochs_per_sample = int(args['--epochs_per_sample'])
        output_dir = 'outputs'
        history = []
        D_costs_train = []
        D_wasses_train = []
        D_costs_valid = []
        D_wasses_valid = []
        G_costs = []
        BATCH_NUM=350
        epochs=1800
        start = time.time()
        self.LOGGER_audio.info('Starting training...EPOCHS={}, BATCH_SIZE={}, BATCH_NUM={}'.format(epochs, self.image_batch_size, BATCH_NUM))
        self.LOGGER_image.info('Starting training...EPOCHS={}, BATCH_SIZE={}, BATCH_NUM={}'.format(epochs, self.audio_batch_size, BATCH_NUM))




        # training loop

        def sample_fake_image_batch(batch_size,real_img):
            return generator.sample_images(batch_size,real_img)

        def sample_fake_video_batch(batch_size,real_video):
            return generator.sample_videos(batch_size,real_video)

        def init_logs():
            return {'l_gen': 0, 'l_image_dis': 0, 'l_video_dis': 0}

        batch_num = 0

        logs = init_logs()

        start_time = time.time()
        epoch=0
        gc.collect()
        while True:
            gc.collect()
            epoch=epoch+1
            self.LOGGER_audio.info("{} Epoch: {}/{}".format(time_since(start), epoch, epochs))
            D_cost_train_epoch = []
            D_wass_train_epoch = []
            D_cost_valid_epoch = []
            D_wass_valid_epoch = []
            G_cost_epoch = []

            self.LOGGER_image.info("{} Epoch: {}/{}".format(time_since(start), epoch, epochs))

            D2_cost_train_epoch = []
            D2_wass_train_epoch = []
            D2_cost_valid_epoch = []
            D2_wass_valid_epoch = []
            G2_cost_epoch = []

            for i in range(1, BATCH_NUM+1):
                 # sample real data
                sample_real_batch = self.sample_real_image_batch()
                batch_real_audio=sample_real_batch['audio'].cpu()
                batch_image = Variable(sample_real_batch['images'], requires_grad=False)
                # print(batch_image.shape)
                batch_image=batch_image.cuda()

                for p in netD.parameters():
                    p.requires_grad = True

                one = torch.Tensor([1]).float()
                neg_one = one * -1
                if self.use_cuda:
                    one = one.cuda()
                    neg_one = neg_one.cuda()
                # (1) Train Discriminator

                for iter_dis in range(1):
                    netD.zero_grad()

                    z = nn.init.normal(torch.Tensor(self.image_batch_size, 512))
                    if self.use_cuda:
                     z = z.cuda()
                    z = Variable(z)

                    real_data_Var = numpy_to_var(batch_real_audio, self.use_cuda)
                    # print(batch_image.shape)
                    # print(type(batch_image))
                    batch_image = batch_image.cuda()


                    real_img_Var = Variable(batch_image)



                    # a) compute loss contribution from real training data
                    D_real = netD(real_data_Var)
                    D_real = D_real.mean()
                    #print('D_real',D_real)  # avg loss
                    D_real.backward(neg_one)  # loss * -1

                    #print('real_img_Var',real_img_Var.shape)
                    D_real_img = ImageDiscriminatorModel(real_img_Var)
                    D_real_img = D_real_img.mean()  # avg loss
                    #print('D_real_img',D_real_img)
                    D_real_img=Variable(D_real_img.data,requires_grad=True)#Added
                    D_real_img.backward(neg_one)  # loss * -1


                    # b) compute loss contribution from generated data, then backprop.
                    features=ImageModel(batch_image)
                    fk_audio=netG(z, features)
                    fk_audio = autograd.Variable(fk_audio.data)
                    #print(fk_audio.shape)(16, 1, 16384)

                    D_fake = netD(fk_audio)
                    D_fake = D_fake.mean()
                    D_fake.backward(one)


                    #print(batch_real_audio.shape)#16*16384
                    batch_real_audio=batch_real_audio.unsqueeze(1)
                    #print(batch_real_audio.shape)#16*1*16384
                    # print(fk_audio.shape)
                    # print(type(fk_audio))
                    # print(batch_real_audio.shape)
                    # print(type(batch_real_audio))
                    # print('audio_encoder',audio_encoder)
                    audio_features=audio_encoder(batch_real_audio)
                    #print('audio_features',audio_features.shape)
                    #audio_features=audio_encoder(fk_audio)
                    audio_features=audio_features.unsqueeze(2).unsqueeze(3)
                    fk_image=ImageGeneratorModel(audio_features)
                    fk_image = autograd.Variable(fk_image.data)
                    D_fake_image = ImageDiscriminatorModel(fk_image)
                    D_fake_image = D_fake_image.mean()
                    D_fake_image=Variable(D_fake_image.data,requires_grad=True)#Added
                    D_fake_image.backward(one)
                    #print('real_img_Var',type(real_img_Var.data))#16x3x224x224








                    # c) compute gradient penalty and backprop
                    gradient_penalty = calc_gradient_penalty(netD, real_data_Var.data,
                                                            fk_audio.data, self.image_batch_size, lmbda,
                                                            use_cuda=self.use_cuda)
                    gradient_penalty.backward(one)



################################# 16*3*224*224 to 16*3*64*64 or do nn.AvgPool2d
                    LRTrans = transforms.Compose([
                    transforms.Scale(64 , Image.BICUBIC),
                    transforms.ToTensor(),
                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                    ])
                    real_img_Var_64=torch.zeros([16,3,64,64])
                    for j in range(16):
                        #print('j',j)
                        real_img_Var_64[j]=torch.FloatTensor(LRTrans(Image.fromarray(
                    real_img_Var.data.cpu().mul(0.5).add(0.5).mul(255).byte()[j].transpose(0, 2).transpose(0, 1).numpy())).numpy())
                    #print('testing',real_img_Var.data.cpu().mul(0.5).add(0.5).mul(255).byte()[j].shape)
                    #print("real_img_Var_64",real_img_Var_64.shape)
                    real_img_Var_64=real_img_Var_64.cuda()
######################################
                    gradient_penalty_2 = calc_gradient_penalty_2(ImageDiscriminatorModel, real_img_Var_64.data,
                                                            fk_image.data, self.audio_batch_size, lmbda,
                                                            use_cuda=self.use_cuda)
                    gradient_penalty_2.backward(one)





                    # Compute cost * Wassertein loss..
                    D_cost_train = D_fake - D_real + gradient_penalty
                    D_wass_train = D_real - D_fake


                    D2_cost_train = D_fake_image - D_real_img + gradient_penalty_2
                    D2_wass_train = D_real_img - D_fake_image

                    # Update gradient of discriminator.
                    optimizerD.step()

                    optimizerD_image.step()

                    #############################
                    # (2) Compute Valid data
                    #############################
                    netD.zero_grad()
                    ImageDiscriminatorModel.zero_grad()
                    batch_real_audio=batch_real_audio.squeeze(1)
                    valid_data_Var = numpy_to_var(batch_real_audio, self.use_cuda)
                    D_real_valid = netD(valid_data_Var)
                    D_real_valid = D_real_valid.mean()  # avg loss

                    #valid_data_Var_2 = numpy_to_var(batch_image, self.use_cuda)# can substitute this with below two lines
                    batch_image = batch_image.cuda()
                    valid_data_Var_2 = Variable(batch_image)




                    D2_real_valid = ImageDiscriminatorModel(valid_data_Var_2)
                    D2_real_valid = D2_real_valid.mean()  # avg loss

                    # b) compute loss contribution from generated data, then backprop.
                    fake_valid = netG(z, features)
                    D_fake_valid = netD(fake_valid)
                    D_fake_valid = D_fake_valid.mean()

                    fake_valid_2 = ImageGeneratorModel(audio_features)
                    D2_fake_valid = ImageDiscriminatorModel(fake_valid_2)
                    D2_fake_valid = D2_fake_valid.mean()

                    # c) compute gradient penalty and backprop
                    gradient_penalty_valid = calc_gradient_penalty(netD, valid_data_Var.data,
                                                                fake_valid.data, self.image_batch_size, lmbda,
                                                                use_cuda=self.use_cuda)

                    ################################# 16*3*224*224 to 16*3*64*64 or do nn.AvgPool2d
                    LRTrans = transforms.Compose([
                    transforms.Scale(64 , Image.BICUBIC),
                    transforms.ToTensor(),
                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                    ])
                    valid_data_Var_2_64=torch.zeros([16,3,64,64])
                    for j in range(16):
                        valid_data_Var_2_64[j]=torch.FloatTensor(LRTrans(Image.fromarray(
                    valid_data_Var_2.data.cpu().mul(0.5).add(0.5).mul(255).byte()[j].transpose(0, 2).transpose(0, 1).numpy())).numpy())
                    #print('testing',real_img_Var.data.cpu().mul(0.5).add(0.5).mul(255).byte()[j].shape)
                    #print("valid_data_Var_2_64",valid_data_Var_2_64.shape)
                    valid_data_Var_2_64=valid_data_Var_2_64.cuda()
######################################

                    
                    
                    
                    
                    
                    
                    
                    
                    gradient_penalty_valid_2 = calc_gradient_penalty_2(ImageDiscriminatorModel, valid_data_Var_2_64.data,
                                                                fake_valid_2.data, self.audio_batch_size, lmbda,
                                                                use_cuda=self.use_cuda)



                    # Compute metrics and record in batch history.
                    D_cost_valid = D_fake_valid - D_real_valid + gradient_penalty_valid
                    D_wass_valid = D_real_valid - D_fake_valid

                    D2_cost_valid = D2_fake_valid - D2_real_valid + gradient_penalty_valid_2
                    D2_wass_valid = D2_real_valid - D2_fake_valid

                    if self.use_cuda:
                        D_cost_train = D_cost_train.cpu()
                        D_wass_train = D_wass_train.cpu()
                        D_cost_valid = D_cost_valid.cpu()
                        D_wass_valid = D_wass_valid.cpu()

                        D2_cost_train = D2_cost_train.cpu()
                        D2_wass_train = D2_wass_train.cpu()
                        D2_cost_valid = D2_cost_valid.cpu()
                        D2_wass_valid = D2_wass_valid.cpu()

                    # Record costs
                    D_cost_train_epoch.append(D_cost_train.data.numpy())
                    D_wass_train_epoch.append(D_wass_train.data.numpy())
                    D_cost_valid_epoch.append(D_cost_valid.data.numpy())
                    D_wass_valid_epoch.append(D_wass_valid.data.numpy())

                    D2_cost_train_epoch.append(D2_cost_train.data.numpy())
                    D2_wass_train_epoch.append(D2_wass_train.data.numpy())
                    D2_cost_valid_epoch.append(D2_cost_valid.data.numpy())
                    D2_wass_valid_epoch.append(D2_wass_valid.data.numpy())



                    #############################
                # (3) Train Generator
                #############################
                # Prevent discriminator update.
                for p in netD.parameters():
                    p.requires_grad = False

                for p in ImageDiscriminatorModel.parameters():
                    p.requires_grad = False

                # Reset generator gradients
                netG.zero_grad()
                fk_audio=netG(z,features)
                # fake = autograd.Variable(fk_img.data)
                # fake = netG(fk_img2)
                G = netD(fk_audio)
                G = G.mean()
                # print('audio_cond',audio_cond.shape)

                # Update gradients.
                G.backward(neg_one)
                G_cost = -G

                optimizerG.step()
                opt_generator.step()


                ImageGeneratorModel.zero_grad()
                fk_img=ImageGeneratorModel(audio_features)
                # fake = autograd.Variable(fk_img.data)
                # fake = netG(fk_img2)
                G2 = ImageDiscriminatorModel(fk_img)
                G2 = G2.mean()
                # print('audio_cond',audio_cond.shape)

                # Update gradients.
                G2.backward(neg_one)
                G2_cost = -G2

                optimizerG_Image.step()
                opt_encoder.step()





                # Record costs
                if self.use_cuda:
                    G_cost = G_cost.cpu()
                G_cost_epoch.append(G_cost.data.numpy())

                if i % (BATCH_NUM //5 ) == 0:
                    self.LOGGER_audio.info("{} Epoch={} Batch: {}/{} D_c:{:.4f} | D_w:{:.4f} | G:{:.4f}".format(time_since(start), epoch,
                                                                                                    i, BATCH_NUM,
                                                                                                    D_cost_train.data.numpy(),
                                                                                                    D_wass_train.data.numpy(),
                                                                                                    G_cost.data.numpy()))




                if self.use_cuda:
                    G2_cost = G2_cost.cpu()
                G2_cost_epoch.append(G2_cost.data.numpy())

                if i % (BATCH_NUM //5 ) == 0:
                    self.LOGGER_image.info("{} Epoch={} Batch: {}/{} D2_c:{:.4f} | D2_w:{:.4f} | G2:{:.4f}".format(time_since(start), epoch,
                                                                                                    i, BATCH_NUM,
                                                                                                    D2_cost_train.data.numpy(),
                                                                                                    D2_wass_train.data.numpy(),
                                                                                                    G2_cost.data.numpy()))



            # Save the average cost of batches in every epoch.
            D_cost_train_epoch_avg = sum(D_cost_train_epoch) / float(len(D_cost_train_epoch))
            D_wass_train_epoch_avg = sum(D_wass_train_epoch) / float(len(D_wass_train_epoch))
            D_cost_valid_epoch_avg = sum(D_cost_valid_epoch) / float(len(D_cost_valid_epoch))
            D_wass_valid_epoch_avg = sum(D_wass_valid_epoch) / float(len(D_wass_valid_epoch))
            G_cost_epoch_avg = sum(G_cost_epoch) / float(len(G_cost_epoch))

            D_costs_train.append(D_cost_train_epoch_avg)
            D_wasses_train.append(D_wass_train_epoch_avg)
            D_costs_valid.append(D_cost_valid_epoch_avg)
            D_wasses_valid.append(D_wass_valid_epoch_avg)
            G_costs.append(G_cost_epoch_avg)




            # Save the average cost of batches in every epoch.
            D2_cost_train_epoch_avg = sum(D2_cost_train_epoch) / float(len(D2_cost_train_epoch))
            D2_wass_train_epoch_avg = sum(D2_wass_train_epoch) / float(len(D2_wass_train_epoch))
            D2_cost_valid_epoch_avg = sum(D2_cost_valid_epoch) / float(len(D2_cost_valid_epoch))
            D2_wass_valid_epoch_avg = sum(D2_wass_valid_epoch) / float(len(D2_wass_valid_epoch))
            G2_cost_epoch_avg = sum(G2_cost_epoch) / float(len(G2_cost_epoch))

            D2_costs_train.append(D2_cost_train_epoch_avg)
            D2_wasses_train.append(D2_wass_train_epoch_avg)
            D2_costs_valid.append(D2_cost_valid_epoch_avg)
            D2_wasses_valid.append(D2_wass_valid_epoch_avg)
            G2_costs.append(G2_cost_epoch_avg)


            self.LOGGER_audio.info("{} D_cost_train:{:.4f} | D_wass_train:{:.4f} | D_cost_valid:{:.4f} | D_wass_valid:{:.4f} | "
                        "G_cost:{:.4f}".format(time_since(start),
                                            D_cost_train_epoch_avg,
                                            D_wass_train_epoch_avg,
                                            D_cost_valid_epoch_avg,
                                            D_wass_valid_epoch_avg,
                                            G_cost_epoch_avg))





            self.LOGGER_image.info("{} D2_cost_train:{:.4f} | D2_wass_train:{:.4f} | D2_cost_valid:{:.4f} | D2_wass_valid:{:.4f} | "
                        "G2_cost:{:.4f}".format(time_since(start),
                                            D2_cost_train_epoch_avg,
                                            D2_wass_train_epoch_avg,
                                            D2_cost_valid_epoch_avg,
                                            D2_wass_valid_epoch_avg,
                                            G2_cost_epoch_avg))



                # Generate audio samples.
            if epoch % epochs_per_sample == 0:
                self.LOGGER_audio.info("Generating samples...")

                self.LOGGER_image.info("Generating image samples...")
                # batch_real_image_val,image_enumerator=sample_real_image_batch(image_enumerator)
                # batch_real_audio_val=batch_real_image_val['audio'].cpu()
                # sample_test_Var = numpy_to_var(batch_real_audio_val, cuda)
                torch.save(ImageModel, os.path.join(self.log_folder, '%05d_ImageModel.pytorch' % epoch))
                torch.save(netG, os.path.join(self.log_folder, '%05d_netG.pytorch' % epoch))
                torch.save(netD, os.path.join(self.log_folder, 'I%05d_netD.pytorch' % epoch))


                torch.save(audio_encoder, os.path.join(self.log_folder, '%05d_audio_encoder.pytorch' % epoch))
                torch.save(ImageGeneratorModel, os.path.join(self.log_folder, '%05d_ImageGeneratorModel.pytorch' % epoch))
                torch.save(ImageDiscriminatorModel, os.path.join(self.log_folder, 'I%05d_ImageDiscriminatorModel.pytorch' % epoch))


                for iii in range(1, 2):
                     # sample real data
                    sample_test_batch = self.sample_test_image_batch()
                    batch_test_audio=sample_test_batch['audio'].cpu()
                    batch_image_test = Variable(sample_test_batch['images'], requires_grad=False)
                    # print(batch_image.shape)
                    batch_image_test=batch_image_test.cuda()
                    features_test=ImageModel(batch_image)
                    z = nn.init.normal(torch.Tensor(self.image_batch_size, 512))
                    if  self.use_cuda:
                        z = z.cuda()
                    z = Variable(z)
                    sample_out = netG(z, features_test)
                    if  self.use_cuda:
                        sample_out = sample_out.cpu()
                    sample_out = sample_out.data.numpy()
                    save_samples(sample_out, epoch, output_dir)
                    gc.collect()

