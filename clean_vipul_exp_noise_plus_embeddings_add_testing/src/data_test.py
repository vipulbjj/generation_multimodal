"""
Copyright (C) 2017 NVIDIA Corporation.  All rights reserved.
Licensed under the CC BY-NC-ND 4.0 license (https://creativecommons.org/licenses/by-nc-nd/4.0/legalcode).
"""

import os
import tqdm
import pickle
import numpy as np
import torch.utils.data
from torchvision.datasets import ImageFolder
import PIL

from numpy import random
import librosa
from pydub import AudioSegment


class VideoFolderDataset(torch.utils.data.Dataset):
    def __init__(self, folder, cache, min_len=20):
        dataset = ImageFolder(folder)
        self.total_frames = 0
        self.lengths = []
        self.images = []

        if cache is not None and os.path.exists(cache):
            with open(cache, 'r') as f:
                self.images, self.lengths = pickle.load(f)
        else:
            for idx, (im, categ) in enumerate(
                    tqdm.tqdm(dataset, desc="Counting total number of frames")):
                img_path, _ = dataset.imgs[idx]
                # print('img_path',img_path,idx)
                shorter, longer = min(im.width, im.height), max(im.width, im.height)
                length = longer // shorter
                if length >= min_len:
                    self.images.append((img_path, categ))
                    self.lengths.append(length)

            if cache is not None:
                with open(cache, 'w') as f:
                    pickle.dump((self.images, self.lengths), f)

        self.cumsum = np.cumsum([0] + self.lengths)
        # print('self.cumsum',self.cumsum)
        print "Total number of frames {}".format(np.sum(self.lengths))

    def __getitem__(self, item):
        # print('get item of video caling',item)
        path, label = self.images[item]
        im = PIL.Image.open(path)
        return im, label,path

    def __len__(self):
        return len(self.images)


class ImageDataset(torch.utils.data.Dataset):
    def __init__(self, dataset, audio_data_folder_path,transform=None):
        self.dataset = dataset
        # print('int of inage is calling')

        self.transforms = transform if transform is not None else lambda x: x

        if not os.path.exists(audio_data_folder_path):
            raise Error('The data folder does not exist!')

        # store full paths - not the actual files.
        # all files cannot be loaded up to memory due to its large size.
        # insted, we read from files upon fetching batches (see __getitem__() implementation)
        self.filepaths = [os.path.join(audio_data_folder_path, filename)
                for filename in sorted(os.listdir(audio_data_folder_path))]
        self.num_data = len(self.filepaths)
        # print('NEW apthself.filepaths',self.filepaths)

    def __getitem__(self, item):
        # print('get item of images is calling')
        if item != 0:
            video_id = np.searchsorted(self.dataset.cumsum, item) - 1
            # print(' ithink  yh acalling ho rhi h ',video_id)
            frame_num = item - self.dataset.cumsum[video_id] - 1
        else:
            video_id = 0
            frame_num = 0

        video, target,path = self.dataset[video_id]
        video = np.array(video)

        horizontal = video.shape[1] > video.shape[0]

        if horizontal:
            i_from, i_to = video.shape[0] * frame_num, video.shape[0] * (frame_num + 1)
            frame = video[:, i_from: i_to, ::]
        else:
            i_from, i_to = video.shape[1] * frame_num, video.shape[1] * (frame_num + 1)
            frame = video[i_from: i_to, :, ::]

        if frame.shape[0] == 0:
            print "video {}. From {} to {}. num {}".format(video.shape, i_from, i_to, item)




        def sample_generator(filepath, window_length=16384, fs=16000):
            """
            Audio sample generator
            """
            # print('filepath',filepath)
            # wav_file_pydub = AudioSegment.from_file(filepath)

            # with wav_file_pydub.export('audio.ogg', format='ogg', codec='libvorbis', bitrate='192k') as wav_file:
            #     wav_file.close()

            # ogg_file_librosa = librosa.load('audio.ogg')
            # print('filepath',filepath)
            # librosa.cache.clear()
            audio_data, _ = librosa.load(filepath, sr=fs)

            # Clip magnitude
            max_mag = np.max(np.abs(audio_data))
            if max_mag > 1:
                audio_data /= max_mag

            # Pad audio to >= window_length.
            audio_len = len(audio_data)
            if audio_len < window_length:
                pad_length = window_length - audio_len
                left_pad = pad_length // 2
                right_pad = pad_length - left_pad

                audio_data = np.pad(audio_data, (left_pad, right_pad), mode='constant')
                audio_len = len(audio_data)

            while True:
                if audio_len == window_length:
                    # If we only have a single 1*window_length audio, just yield.
                    sample = audio_data
                else:
                    # Sample a random window from the audio
                    start_idx = np.random.randint(0, (audio_len - window_length) // 2)
                    end_idx = start_idx + window_length
                    sample = audio_data[start_idx:end_idx]

                sample = sample.astype('float32')
                assert not np.any(np.isnan(sample))

                return  sample
         # get item for specified index)
        audio_sample=sample_generator(self.filepaths[video_id])
        return {"images": self.transforms(frame), "categories": target,"audio":audio_sample}

    def __len__(self):
        return self.dataset.cumsum[-1]


class VideoDataset(torch.utils.data.Dataset):
    def __init__(self, dataset, video_length, every_nth=1, transform=None):
        self.dataset = dataset
        self.video_length = video_length
        self.every_nth = every_nth
        self.transforms = transform if transform is not None else lambda x: x

    def __getitem__(self, item):
        video, target,path = self.dataset[item]
        video = np.array(video)
        # print('video ki id',item)

        horizontal = video.shape[1] > video.shape[0]
        shorter, longer = min(video.shape[0], video.shape[1]), max(video.shape[0], video.shape[1])
        video_len = longer // shorter

        # videos can be of various length, we randomly sample sub-sequences
        if video_len > self.video_length * self.every_nth:
            needed = self.every_nth * (self.video_length - 1)
            gap = video_len - needed
            start = 0 if gap == 0 else np.random.randint(0, gap, 1)[0]
            subsequence_idx = np.linspace(start, start + needed, self.video_length, endpoint=True, dtype=np.int32)
        elif video_len >= self.video_length:
            subsequence_idx = np.arange(0, self.video_length)
        else:
            raise Exception("Length is too short id - {}, len - {}").format(self.dataset[item], video_len)

        frames = np.split(video, video_len, axis=1 if horizontal else 0)
        selected = np.array([frames[s_id] for s_id in subsequence_idx])

        return {"images": self.transforms(selected), "categories": target}

    def __len__(self):
        return len(self.dataset)


class ImageSampler(torch.utils.data.Dataset):
    def __init__(self, dataset, transform=None):
        self.dataset = dataset
        self.transforms = transform

    def __getitem__(self, index):
        result = {}
        for k in self.dataset.keys:
            result[k] = np.take(self.dataset.get_data()[k], index, axis=0)

        if self.transforms is not None:
            for k, transform in self.transforms.iteritems():
                result[k] = transform(result[k])

        return result

    def __len__(self):
        return self.dataset.get_data()[self.dataset.keys[0]].shape[0]


class VideoSampler(torch.utils.data.Dataset):
    def __init__(self, dataset, video_length, every_nth=1, transform=None):
        self.dataset = dataset
        self.video_length = video_length
        self.unique_ids = np.unique(self.dataset.get_data()['video_ids'])
        self.every_nth = every_nth
        self.transforms = transform

    def __getitem__(self, item):
        result = {}
        ids = self.dataset.get_data()['video_ids'] == self.unique_ids[item]
        ids = np.squeeze(np.squeeze(np.argwhere(ids)))
        for k in self.dataset.keys:
            result[k] = np.take(self.dataset.get_data()[k], ids, axis=0)

        subsequence_idx = None
        print result[k].shape[0]

        # videos can be of various length, we randomly sample sub-sequences
        if result[k].shape[0] > self.video_length:
            needed = self.every_nth * (self.video_length - 1)
            gap = result[k].shape[0] - needed
            start = 0 if gap == 0 else np.random.randint(0, gap, 1)[0]
            subsequence_idx = np.linspace(start, start + needed, self.video_length, endpoint=True, dtype=np.int32)
        elif result[k].shape[0] == self.video_length:
            subsequence_idx = np.arange(0, self.video_length)
        else:
            print "Length is too short id - {}, len - {}".format(self.unique_ids[item], result[k].shape[0])

        if subsequence_idx:
            for k in self.dataset.keys:
                result[k] = np.take(result[k], subsequence_idx, axis=0)
        else:
            print result[self.dataset.keys[0]].shape

        if self.transforms is not None:
            for k, transform in self.transforms.iteritems():
                result[k] = transform(result[k])

        return result

    def __len__(self):
        return len(self.unique_ids)

