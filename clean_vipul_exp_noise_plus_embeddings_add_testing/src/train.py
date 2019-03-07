"""
Copyright (C) 2017 NVIDIA Corporation.  All rights reserved.
Licensed under the CC BY-NC-ND 4.0 license (https://creativecommons.org/licenses/by-nc-nd/4.0/legalcode).

Sergey Tulyakov, Ming-Yu Liu, Xiaodong Yang, Jan Kautz, MoCoGAN: Decomposing Motion and Content for Video Generation
https://arxiv.org/abs/1707.04993

Usage:
    train.py [options] <dataset> <dataset_test> <log_folder>

Options:
    --image_dataset=<path>          specifies a separate dataset to train for images [default: ]
    --image_batch=<count>           number of images in image batch [default: 10]
    --video_batch=<count>           number of videos in video batch [default: 3]

    --image_size=<int>              resize all frames to this size [default: 224]

    --use_infogan                   when specified infogan loss is used

    --use_categories                when specified ground truth categories are used to
                                    train CategoricalVideoDiscriminator

    --use_noise                     when specified instance noise is used
    --noise_sigma=<float>           when use_noise is specified, noise_sigma controls
                                    the magnitude of the noise [default: 0]

    --image_discriminator=<type>    specifies image disciminator type (see models.py for a
                                    list of available models) [default: PatchImageDiscriminator]

    --video_discriminator=<type>    specifies video discriminator type (see models.py for a
                                    list of available models) [default: CategoricalVideoDiscriminator]

    --video_length=<len>            length of the video [default: 16]
    --print_every=<count>           print every iterations [default: 1]
    --n_channels=<count>            number of channels in the input data [default: 3]
    --every_nth=<count>             sample training videos using every nth frame [default: 4]
    --batches=<count>               specify number of batches to train [default: 100000]

    --dim_z_content=<count>         dimensionality of the content input, ie hidden space [default: 50]
    --dim_z_motion=<count>          dimensionality of the motion input [default: 10]
    --dim_z_category=<count>        dimensionality of categorical input [default: 6]


    --model_size=<int>        model_size [default: 64]
    --shift_factor=<int>        shift_factor [default: 2]
    --post_proc_filt_len=<int>       post_proc_filt_len [default: 512]
    --alpha=<float>       alpha [default: 0.2]
    --batch_size=<int>       batch_size [default: 16]
    --num_epochs=<int>       num_epochs [default: 180]
    --ngpus=<int>       ngpus [default: 4]
    --latent_dim=<int>       latent_dim [default: 100]
    --epochs_per_sample=<int>       epochs_per_sample [default: 1]
    --sample_size=<int>       sample_size [default: 10]
    --lmbda=<float>       lmbda [default: 10.0]
    --learning_rate=<float>       learning_rate [default: 1e-4]
    --beta1=<float>       beta1 [default: 0.5]
    --beta2=<float>       beta2 [default: 0.9]
    --verbose=<count>       verbose [default:'store_true']
    --audio_dir=<path>       audio_dir [default:'../try/']
    --output_dir=<path>       output_dir [default:'output']

"""

#!/usr/bin/env python -W ignore::DeprecationWarning

##Video
from __future__ import print_function
import os
import docopt
import PIL
import functools
from trainers import Trainer
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
import models
import data

#Audio
import torch
from torch import autograd
from torch import optim
import json
from utils import save_samples
import numpy as np
import pprint
import pickle
import datetime
from wavegan import *
from utils import *
from logger import *
from torch.utils.data import DataLoader
from torchvision import transforms
import PIL
import data
import data_test
import warnings
warnings.filterwarnings("ignore", category=FutureWarning) 

# def fxn():
#     warnings.warn("deprecated", DeprecationWarning)

# with warnings.catch_warnings():
#     warnings.simplefilter("ignore")
#     fxn()



def build_discriminator(type, **kwargs):
    discriminator_type = getattr(models, type)

    if 'Categorical' not in type and 'dim_categorical' in kwargs:
        kwargs.pop('dim_categorical')

    return discriminator_type(**kwargs)


def video_transform(video, image_transform):
    vid = []
    for im in video:
        vid.append(image_transform(im))

    vid = torch.stack(vid).permute(1, 0, 2, 3)

    return vid


if __name__ == "__main__":
    args = docopt.docopt(__doc__)
    # print args
 ##=========#Audio part ==========================================
    cuda = True if torch.cuda.is_available() else False
    # =============Logger===============
    LOGGER = logging.getLogger('wavegan')
    LOGGER.setLevel(logging.DEBUG)
    LOGGER.info('Initialized logger.')
    init_console_logger(LOGGER)
    # =============Parameters===============
    epochs = int(args['--num_epochs'])
    batch_size = int(args['--batch_size'])
    latent_dim = int(args['--latent_dim'])
    ngpus = int(args['--ngpus'])
    model_size = int(args['--model_size'])
    model_dir = make_path(os.path.join(args['--output_dir'],
                                    datetime.datetime.now().strftime("%Y%m%d%H%M%S")))
    # args['model_dir'] = model_dir
    # save samples for every N epochs.
    epochs_per_sample = int(args['--epochs_per_sample'])
    # gradient penalty regularization factor.
    lmbda = float(args['--lmbda'])
    # Dir
    # audio_dir = args['audio_dir']
    audio_dir ='../data/audio_dataset/audio_combined'
    audio_dir_test ='../data/audio_dataset/audio_combined_test'
    print('audio_dir',audio_dir)
    output_dir = 'outputs'

############################################################################

    n_channels = int(args['--n_channels'])
    image_transforms = transforms.Compose([
        PIL.Image.fromarray,
        transforms.Scale(int(args["--image_size"])),
        transforms.ToTensor(),
        lambda x: x[:n_channels, ::],
        transforms.Normalize((0.5, 0.5, .5), (0.5, 0.5, 0.5)),
    ])

    video_transforms = functools.partial(video_transform, image_transform=image_transforms)

    video_length = int(args['--video_length'])
    image_batch = int(args['--image_batch'])
    video_batch = int(args['--video_batch'])

    dim_z_content = int(args['--dim_z_content'])
    dim_z_motion = int(args['--dim_z_motion'])
    dim_z_category = int(args['--dim_z_category'])

    dataset = data.VideoFolderDataset(args['<dataset>'], cache=os.path.join(args['<dataset>'], 'local.db'))
    image_dataset = data.ImageDataset(dataset,audio_dir, image_transforms)
    image_loader = DataLoader(image_dataset, batch_size=image_batch, drop_last=True, num_workers=6, shuffle=True)
    print('args[<dataset>',args['<dataset>'])
    print('args[<dataset_test>',args['<dataset_test>'])
    dataset_test = data_test.VideoFolderDataset(args['<dataset_test>'], cache=os.path.join(args['<dataset_test>'], 'local_test.db'))
    image_dataset_test = data_test.ImageDataset(dataset_test,audio_dir_test, image_transforms)
    image_loader_test = DataLoader(image_dataset_test, batch_size=image_batch, drop_last=True, num_workers=6, shuffle=False)


    ImageModel = models.ImageConvNet().cuda()

    audio_encoder = WaveGANDiscriminator512(model_size=model_size, ngpus=ngpus)

    # video_dataset = data.VideoDataset(dataset, 16, 2, video_transforms)
    # video_loader = DataLoader(video_dataset, batch_size=video_batch, drop_last=True, num_workers=6, shuffle=True)

    # generator = models.VideoGenerator(n_channels, dim_z_content, dim_z_category, dim_z_motion, video_length)

    # image_discriminator = build_discriminator(args['--image_discriminator'], n_channels=n_channels,
    #                                           use_noise=args['--use_noise'], noise_sigma=float(args['--noise_sigma']))

    # video_discriminator = build_discriminator(args['--video_discriminator'], dim_categorical=dim_z_category,
    #                                           n_channels=n_channels, use_noise=args['--use_noise'],
    #                                           noise_sigma=float(args['--noise_sigma']))


    ##Audio part
    # netG = WaveGANGenerator(model_size=model_size, ngpus=ngpus, latent_dim=latent_dim, upsample=True)
    # netG=WaveGANGenerator()
    netG=WaveGANGenerator(model_size=model_size, ngpus=ngpus, latent_dim=512, upsample=True)
    netD = WaveGANDiscriminator(model_size=model_size, ngpus=ngpus)


    ImageGeneratorModel = models.ImageGeneratorDC().cuda()
    ImageDiscriminatorModel = models.ImageDiscriminatorDC().cuda()




    if cuda:
        netG = torch.nn.DataParallel(netG).cuda()
        netD = torch.nn.DataParallel(netD).cuda()
        audio_encoder = torch.nn.DataParallel(audio_encoder).cuda()
     # Save config.
    LOGGER.info('Saving configurations...')
    config_path = os.path.join(model_dir, 'config.json')
    with open(config_path, 'w') as f:
        json.dump(args, f)
    # Load data.
    LOGGER.info('Loading audio data...')



    # if torch.cuda.is_available():
        # generator.cuda()
        # image_discriminator.cuda()
        # video_discriminator.cuda()


#need other logger for image part
    trainer = Trainer(image_loader,image_loader_test,
                      int(args['--print_every']),
                      int(args['--batches']),
                      args['<log_folder>'],LOGGER,LOGGER,
                      use_cuda=torch.cuda.is_available(),
                      use_infogan=args['--use_infogan'],
                      use_categories=args['--use_categories'])

    trainer.train(ImageModel,netG,netD,audio_encoder,ImageGeneratorModel,ImageDiscriminatorModel, args)



