import torch
from torch.utils.data import Dataset
import h5py
import json
import os
from PIL import Image
from matplotlib.pyplot import imread
import numpy as np


class CaptionDataset(Dataset):
    """
    A PyTorch Dataset class to be used in a PyTorch DataLoader to create batches.
    """

    def __init__(self, data_folder, split, transform=None):
        """
        :param data_folder: folder where data files are stored
        :param split: split, one of 'TRAIN', 'VAL', or 'TEST'
        :param transform: image transform pipeline
        """
        self.split = split
        assert self.split in {'TRAIN', 'VAL', 'TEST'}

        # Open hdf5 file where images are stored
        # self.imgs = h5py.File(os.path.join(data_folder, self.split + '_FILENAMES' + '.json'), 'r')
        # self.imgs = self.h['images']

        # Load image filenames (completely into memory)
        with open(os.path.join(data_folder, self.split + '_FILENAMES' + '.json'), 'r') as j:
            self.filenames = json.load(j)

        # Captions per image
        self.cpi = 1 # self.h.attrs['captions_per_image']

        # Load encoded captions (completely into memory)
        with open(os.path.join(data_folder, self.split + '_CAPTIONS' + '.json'), 'r') as j:
            self.captions = json.load(j)

        # Load caption lengths (completely into memory)
        with open(os.path.join(data_folder, self.split + '_CAPLENS' + '.json'), 'r') as j:
            self.caplens = json.load(j)

        # PyTorch transformation pipeline for the image (normalizing, etc.)
        self.transform = transform

        # Total number of datapoints
        self.dataset_size = len(self.captions)

    def __getitem__(self, i):
        # Remember, the Nth caption corresponds to the (N // captions_per_image)th image
        # img = torch.FloatTensor(self.imgs[i // self.cpi] / 255.)
        img = Image.open(self.filenames[i]).convert('RGB')
        # img = img.resize((224, 224))
        # img = np.array(img) 
        
        # img = imread(self.filenames[i])
        # if len(img.shape) == 2:
        #     img = img[:, :, np.newaxis]
        #     img = np.concatenate([img, img, img], axis=2)
        # img = img.transpose(2, 0, 1)
        # img = imresize(img, (256, 256))
        # assert img.shape == (3, 224, 224)
        # assert np.max(img) <= 255
        
        # img = torch.FloatTensor(img / 255.) 
        
        if self.transform is not None:
            img = self.transform(img)

        assert img.shape == (3, 224, 224)
        
        # batch_size, num_channels, height, width = img.size()    
        # img = torch.autograd.Variable(img.view(-1, num_channels, height, width).cuda())

        caption = torch.LongTensor(self.captions[i])

        caplen = torch.LongTensor([self.caplens[i]])

        if self.split is 'TRAIN':
            return img, caption, caplen
        else:
            # For validation of testing, also return all 'captions_per_image' captions to find BLEU-4 score
            all_captions = torch.LongTensor(
                self.captions[((i // self.cpi) * self.cpi):(((i // self.cpi) * self.cpi) + self.cpi)])
            return img, caption, caplen, all_captions

    def __len__(self):
        return self.dataset_size
