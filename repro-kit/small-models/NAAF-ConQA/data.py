# -------------------------------------------------------------------------------------
# Negative-Aware Attention Framework for Image-Text Matching  implementation based on SCAN
# https://github.com/CrossmodalGroup/NAAF
# "Negative-Aware Attention Framework for Image-Text Matching"
# Kun Zhang, Zhendong Mao, Quan Wang, Yongdong Zhang
#
# Writen by Kun Zhang, 2022
# -------------------------------------------------------------------------------------
"""Data provider"""


import torch
import torch.utils.data as data
import torchvision.transforms as transforms
import os
from PIL import Image
import numpy as np
import json as jsonmod
import json
import nltk
import pickle

class PrecompDataset(data.Dataset):
    """
    Load precomputed captions and image features
    Possible options: f30k_precomp, coco_precomp
    """

    def __init__(self, data_path, experiment, data_split, vocab):
        self.vocab = vocab
        
        self.caption_non = []
        
        if experiment == 'conqa':
            with open(f'{data_path}/conqa-queries.pkl', 'rb') as f:
                conqa_q = pickle.load(f)
            
            for cap in list(conqa_q.values()):
                self.caption_non.append(cap)
        
        elif experiment == 'gpt':
            with open(f'{data_path}/gpt-conqa-queries.pkl', 'rb') as f:
                gpt_conqa_q = pickle.load(f)     
            
            for cap in list(gpt_conqa_q.values()):
                self.caption_non.append(cap)
                
        elif experiment == 'gpt_abs':
            with open(f'{data_path}/gpt-abs-conqa-queries.pkl', 'rb') as f:
                gpt_conqa_q = pickle.load(f)     
            
            for cap in list(gpt_conqa_q.values()):
                self.caption_non.append(cap)
                
        elif experiment == 'gpt_nonabs':
            with open(f'{data_path}/gpt-nonabs-conqa-queries.pkl', 'rb') as f:
                gpt_conqa_q = pickle.load(f)     
            
            for cap in list(gpt_conqa_q.values()):
                self.caption_non.append(cap)                
            
        elif experiment == 'coco':
            with open(f'{data_path}/coco-conqa-queries.pkl', 'rb') as f:
                coco_conqa_q = pickle.load(f)        
            
            for cap in list(coco_conqa_q.values()):
                self.caption_non.append(cap)
                                        
        elif experiment == 'ext_coco':
            with open(f'{data_path}/ext-coco-conqa-queries.pkl', 'rb') as f:
                ext_coco_conqa_q = pickle.load(f)  
            
            for cap in list(ext_coco_conqa_q.values()):
                self.caption_non.append(cap)
                
        elif experiment == 'coco5k':
            with open(f'{data_path}/capts.txt', 'r') as f:
                for line in f:
                    if line.strip()!= '':
                        self.caption_non.append(line.strip())
            self.images = np.load(f'{data_path}/data/coco_precomp/testall_ims.npy')
        
        
        if experiment != 'coco5k':
            # Image features
            self.images = []        
            conqa_small_precomp_features = np.load(f'{data_path}/conqa_small_precomp_features_fit.npz')['imgs']
            
            for i in conqa_small_precomp_features:
                self.images.append(i)
            
            if len(self.caption_non) > len(self.images):
                for i in range(len(self.caption_non) - len(self.images)):
                    self.images.append(self.images[0])
            else:         
                for i in range(len(self.images) - len(self.caption_non)):
                    cap = 'undefined'
                    self.caption_non.append(cap)        
            
            self.images = np.array(self.images)
            
        
        
        self.length = len(self.caption_non)

        # rkiros data has redundancy in images, we divide by 5, 10crop doesn't
        if self.images.shape[0] != self.length:
            self.im_div = 5
        else:
            self.im_div = 1
        # the development set for coco is large and so validation would be slow
        # if data_split == 'dev':
        #     self.length = 5000

    def __getitem__(self, index):
        # handle the image redundancy
        img_id = index // self.im_div
        # img_id = index
        image = torch.Tensor(self.images[img_id])
        caption_non = self.caption_non[index]
        vocab = self.vocab

        tokens = nltk.tokenize.word_tokenize(
            str(caption_non).lower())
        caption_non = []
        caption_non.append(vocab('<start>'))
        caption_non.extend([vocab(token) for token in tokens])
        caption_non.append(vocab('<end>'))

        captions = torch.Tensor(caption_non)
        return image, captions, index, img_id

    def __len__(self):
        return self.length



def collate_fn(data):
    """Build mini-batch tensors from a list of (image, caption) tuples.
    Args:
        data: list of (image, caption) tuple.
            - image: torch tensor of shape (3, 256, 256).
            - caption: torch tensor of shape (?); variable length.

    Returns:
        images: torch tensor of shape (batch_size, 3, 256, 256).
        targets: torch tensor of shape (batch_size, padded_length).
        lengths: list; valid length for each padded caption.
    """
    # Sort a data list by caption length
    data.sort(key=lambda x: len(x[1]), reverse=True)
    images, captions, ids, img_ids = zip(*data)

    # Merge images (convert tuple of 3D tensor to 4D tensor)
    images = torch.stack(images, 0)

    # Merget captions (convert tuple of 1D tensor to 2D tensor)
    lengths = [len(cap) for cap in captions]
    targets = torch.zeros(len(captions), max(lengths)).long()
    for i, cap in enumerate(captions):
        end = lengths[i]
        targets[i, :end] = cap[:end]

    return images, targets, lengths, ids

def get_precomp_loader(data_path, experiment, data_split, vocab, opt, batch_size=100,
                       shuffle=True, num_workers=2):
    """Returns torch.utils.data.DataLoader for custom coco dataset."""
    dset = PrecompDataset(data_path, experiment, data_split, vocab)

    data_loader = torch.utils.data.DataLoader(dataset=dset,
                                              batch_size=batch_size,
                                              shuffle=shuffle,
                                              pin_memory=True,
                                              collate_fn=collate_fn)
    return data_loader


def get_loaders(data_name, vocab, batch_size, workers, opt):
    dpath = os.path.join(opt.data_path, data_name)
    train_loader = get_precomp_loader(dpath, 'train', vocab, opt,
                                      batch_size, True, workers)
    val_loader = get_precomp_loader(dpath, 'dev', vocab, opt,
                                    batch_size, False, workers)
    return train_loader, val_loader


def get_test_loader(split_name, experiment, data_name, vocab, batch_size,
                    workers, opt):
    dpath = opt.data_path
    test_loader = get_precomp_loader(dpath, experiment, split_name, vocab, opt,
                                     batch_size, False, workers)
    return test_loader
